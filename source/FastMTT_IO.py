import numpy as np
import time
import FastMTT
import multiprocessing as mp
import pandas as pd

# Optional dependency: only required for ROOT input
try:
    import uproot  # type: ignore
except Exception:  # pragma: no cover
    uproot = None

# Globalna instancja dla każdego procesu
global_fMTT = None  

# Likelihood attributes that may be set through gj_options
GJ_OPTION_NAMES = ("gj_sigma_angle", "gj_max_pull", "gj_model")

###############################################################################
# Placeholder PV / SV resolution [cm], used when the input has no vertex covariance.
#
# Calibrated on Run3_2022 ggH->tautau (3-prong legs with a refitted SV, tt+mt+et):
# the SV displacement perpendicular to the generated tau direction was fitted versus
# flight length L as sigma^2(L) = s_v^2 + (s_a L)^2. With the beamspot-constrained PV
# (PVBS) this gives s_v = 10 um in the transverse plane and ~17 um along z, and an
# angular term s_a ~ 1.3-1.5 mrad (part of it is rounding of the stored gen direction).
# Only the PV+SV sum is measured this way; the split below is an assumption and only
# affects the correlation between the two legs. The angular term is represented by
# Likelihood.gj_sigma_angle. Replace with event-by-event covariances once available.
###############################################################################
PLACEHOLDER_PV_SIGMA = (5.0e-4, 5.0e-4, 8.0e-4)
PLACEHOLDER_SV_SIGMA = (8.7e-4, 8.7e-4, 15.0e-4)


def placeholder_vertex_covariances(n, scale=1.0):
    """Return (covPV (n, 3, 3), covSV (n, 2, 3, 3)) from the placeholder resolution model."""
    covPV = np.diag(np.square(PLACEHOLDER_PV_SIGMA)) * scale**2
    covSV = np.diag(np.square(PLACEHOLDER_SV_SIGMA)) * scale**2
    return (np.broadcast_to(covPV, (n, 3, 3)).copy(),
            np.broadcast_to(covSV, (n, 2, 3, 3)).copy())


def init_worker(window=False, gj=False, gj_options=None):
    # One FastMTT object for each core
    global global_fMTT
    global_fMTT = FastMTT.FastMTT()
    # Configure likelihood components per worker
    global_fMTT.myLikelihood.enableLikelihoodComponents(window=window, GJ=gj)
    for name, val in (gj_options or {}).items():
        if name not in GJ_OPTION_NAMES:
            raise ValueError(f"Unknown GJ option '{name}', expected one of {GJ_OPTION_NAMES}")
        setattr(global_fMTT.myLikelihood, name, val)

def process_batches_for_worker(args):
    worker_id, worker_batches = args
    # Each core processes its own batches
    global global_fMTT
    results = []
    for batch_data in worker_batches:
        measuredTau, METx, METy, covMET = batch_data
        global_fMTT.run(measuredTau, METx, METy, covMET)
        results.append((global_fMTT.mass, global_fMTT.pt, global_fMTT.tau1pt, global_fMTT.tau2pt))
    return results


def process_single_batch(args):
    batch_id, measuredTau, METx, METy, covMET, vertex = args
    global global_fMTT
    global_fMTT.run(measuredTau, METx, METy, covMET, **(vertex or {}))
    return batch_id, (global_fMTT.mass, global_fMTT.pt, global_fMTT.tau1pt, global_fMTT.tau2pt, global_fMTT.gj_used)

def process_FastMTT(measuredTauLeptons, xMETs, yMETs, covMETs, batch_size=100, num_workers=4, window=False,
                    vertex=None, gj=False, gj_options=None, return_gj_used=False):
    """Run FastMTT in parallel batches.

    vertex: optional dict with keys PV, SV, covPV, covSV, hasSV (arrays with N rows),
            passed to FastMTT.run for the Gottfried-Jackson component (enabled with gj=True).
    gj_options: optional dict of Likelihood attributes, see GJ_OPTION_NAMES.
    return_gj_used: also return an (N, 2) bool array of legs that used the GJ component.
    """
    num_total = len(measuredTauLeptons)
    num_batches = int(np.ceil(num_total / batch_size))

    if num_total == 0:
        empty = (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
        return empty + (np.empty((0, 2), dtype=bool),) if return_gj_used else empty

    print(
        f"FastMTT: n_events={num_total}, batch_size={batch_size}, n_batches={num_batches}, "
        f"workers={num_workers}, window={window}, gj={gj}"
    )

    # Build batch payloads (batch_id keeps original ordering)
    batch_args = []
    for batch_id, start in enumerate(range(0, num_total, batch_size)):
        stop = min(start + batch_size, num_total)
        batch_args.append(
            (
                batch_id,
                measuredTauLeptons[start:stop],
                xMETs[start:stop],
                yMETs[start:stop],
                covMETs[start:stop],
                None if vertex is None else {k: v[start:stop] for k, v in vertex.items()},
            )
        )
    
    start_time = time.time()
    
    # Multiprocessing: process per-batch so the parent can report progress.
    results_by_batch = [None] * num_batches
    processed = 0
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(window, gj, gj_options)) as pool:
        for batch_id, out in pool.imap_unordered(process_single_batch, batch_args, chunksize=1):
            results_by_batch[batch_id] = out
            # out[0] is mass array for this batch
            processed += len(out[0])
            print(f"  processed {processed}/{num_total} events (batch {batch_id + 1}/{num_batches})")

    # Concatenate in the correct original order
    mFast, ptFast, tau1pt, tau2pt, gj_used = zip(*results_by_batch)
    
    end_time = time.time()
    print(f"Processing FastMTT took {end_time - start_time:.2f} seconds")
    
    out = (np.concatenate(mFast, axis=0), np.concatenate(ptFast, axis=0), np.concatenate(tau1pt, axis = 0), np.concatenate(tau2pt, axis = 0))
    if return_gj_used:
        out = out + (np.concatenate(gj_used, axis=0),)
    return out

def read_root_file(file_path, tree_name, branches, entry_stop=None):
    if uproot is None:
        raise ImportError(
            "Reading .root requires 'uproot'. Install it or use .csv/.parquet input."
        )

    with uproot.open(file_path) as file:
        tree = file[tree_name]
        
        data = tree.arrays(branches, library="np", entry_stop=entry_stop)
    
    return data

def load_root_events(file_path, tree_name, branches, entry_stop=None):
    data = read_root_file(file_path, tree_name, branches, entry_stop)

    shape = data["pt_1"].shape

    measuredTauLeptons = np.array([
        [np.full(shape, 3),  data["pt_1"], data["eta_1"], data["phi_1"], data["m_1"], data["dm_1"]],
        [np.full(shape, 1),  data["pt_2"], data["eta_2"], data["phi_2"], data["m_2"], data["dm_2"]]
    ])
    measuredTauLeptons = np.transpose(measuredTauLeptons, (2, 0, 1))

    covMET = np.array([
        [data["metcov00"], data["metcov01"]],
        [data["metcov01"], data["metcov11"]]
    ])
    covMET = np.transpose(covMET, (2, 0, 1))

    METx = data["met"] * np.cos(data["metphi"])
    METy = data["met"] * np.sin(data["metphi"])

    return {
        "measuredTauLeptons": measuredTauLeptons,
        "measuredMETx": METx,
        "measuredMETy": METy,
        "covMET": covMET
    }

def load_events_csv(csv_data):

    df = pd.read_csv(csv_data)

    event_df = df[['H.m', 'H.pt', 'METx', 'METy', 'covXX', 'covXY', 'covYY', 'dm1', 'pt1', 'eta1', 'phi1', 'mass1', 'type1', 'dm2', 'pt2', 'eta2', 'phi2', 'mass2', 'type2']].copy()

    Higgs_mass = event_df.pop('H.m').to_numpy()
    Higgs_pt = event_df.pop('H.pt').to_numpy()
    METx = event_df.pop('METx').to_numpy()
    METy = event_df.pop('METy').to_numpy()
    metcov = event_df[['covXX', 'covXY', 'covXY', 'covYY']].to_numpy()
    event_df.drop(columns=['covXX', 'covXY', 'covYY'], inplace=True)
    metcov = np.reshape(metcov, (len(metcov), 2, 2))

    print('pandas dataframe:\n', event_df)

    events = event_df.to_numpy()
    events = np.reshape(events, (len(events), 2, 6))

    return {"measuredTauLeptons": events, "measuredMETx": METx, "measuredMETy": METy, "covMET": metcov, "Higgs_mass": Higgs_mass, "Higgs_pt": Higgs_pt}


def _first_existing(columns, preferred, fallbacks=()):
    for name in (preferred, *fallbacks):
        if name in columns:
            return name
    return None


def load_events_parquet(parquet_path, columns=None):
    """Load FastMTT inputs from a parquet file.

    Supports two schemas:
    1) CSV-like HiggsDNA toy schema (columns like 'METx', 'covXX', 'pt1', ...)
    2) Ntuple-like schema (columns like 'pt_1', 'eta_1', 'phi_1', 'mass_1' or 'm_1', 'met_pt', ...)
    """

    try:
        df = pd.read_parquet(parquet_path, columns=columns)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet file '{parquet_path}'. "
            "This typically requires 'pyarrow' (recommended) or 'fastparquet'."
        ) from e

    # 1) If it looks like the CSV schema, reuse the CSV loader logic
    if "METx" in df.columns and "covXX" in df.columns and "pt1" in df.columns:
        # Write to a temporary CSV-like in-memory dataframe transformation
        event_df = df[[
            'H.m', 'H.pt', 'METx', 'METy', 'covXX', 'covXY', 'covYY',
            'dm1', 'pt1', 'eta1', 'phi1', 'mass1', 'type1',
            'dm2', 'pt2', 'eta2', 'phi2', 'mass2', 'type2'
        ]].copy()

        Higgs_mass = event_df.pop('H.m').to_numpy()
        Higgs_pt = event_df.pop('H.pt').to_numpy()
        METx = event_df.pop('METx').to_numpy()
        METy = event_df.pop('METy').to_numpy()
        metcov = event_df[['covXX', 'covXY', 'covXY', 'covYY']].to_numpy()
        event_df.drop(columns=['covXX', 'covXY', 'covYY'], inplace=True)
        metcov = np.reshape(metcov, (len(metcov), 2, 2))

        events = event_df.to_numpy()
        events = np.reshape(events, (len(events), 2, 6))

        return {
            "measuredTauLeptons": events,
            "measuredMETx": METx,
            "measuredMETy": METy,
            "covMET": metcov,
            "Higgs_mass": Higgs_mass,
            "Higgs_pt": Higgs_pt,
        }

    # 2) Otherwise try the ntuple-like schema
    cols = set(df.columns)
    b_pt1 = _first_existing(cols, "pt_1")
    b_eta1 = _first_existing(cols, "eta_1")
    b_phi1 = _first_existing(cols, "phi_1")
    b_m1 = _first_existing(cols, "mass_1", ("m_1",))

    b_pt2 = _first_existing(cols, "pt_2")
    b_eta2 = _first_existing(cols, "eta_2")
    b_phi2 = _first_existing(cols, "phi_2")
    b_m2 = _first_existing(cols, "mass_2", ("m_2",))

    b_met = _first_existing(cols, "met_pt", ("met",))
    b_metphi = _first_existing(cols, "met_phi", ("metphi",))

    b_covxx = _first_existing(cols, "met_covXX", ("metcov00",))
    b_covxy = _first_existing(cols, "met_covXY", ("metcov01",))
    b_covyy = _first_existing(cols, "met_covYY", ("metcov11",))

    b_dm1 = _first_existing(cols, "decayModePNet_1", ("decayMode_1", "dm_1", "dm1"))
    b_dm2 = _first_existing(cols, "decayModePNet_2", ("decayMode_2", "dm_2", "dm2"))
    b_type1 = _first_existing(cols, "decay_type_1", ("type_1", "type1"))
    b_type2 = _first_existing(cols, "decay_type_2", ("type_2", "type2"))

    required = [
        b_pt1, b_eta1, b_phi1, b_m1,
        b_pt2, b_eta2, b_phi2, b_m2,
        b_met, b_metphi,
        b_covxx, b_covxy, b_covyy,
    ]
    if any(x is None for x in required):
        missing = [x for x in required if x is None]
        raise ValueError(
            "Unsupported parquet schema for FastMTT input; missing required columns. "
            "Expected columns like pt_*, eta_*, phi_*, mass_*/m_*, met_pt/met, met_phi/metphi, met_covXX/XY/YY."
        )

    pt1 = df[b_pt1].to_numpy(dtype=np.float64)
    eta1 = df[b_eta1].to_numpy(dtype=np.float64)
    phi1 = df[b_phi1].to_numpy(dtype=np.float64)
    m1 = df[b_m1].to_numpy(dtype=np.float64)

    pt2 = df[b_pt2].to_numpy(dtype=np.float64)
    eta2 = df[b_eta2].to_numpy(dtype=np.float64)
    phi2 = df[b_phi2].to_numpy(dtype=np.float64)
    m2 = df[b_m2].to_numpy(dtype=np.float64)

    met = df[b_met].to_numpy(dtype=np.float64)
    metphi = df[b_metphi].to_numpy(dtype=np.float64)

    covxx = df[b_covxx].to_numpy(dtype=np.float64)
    covxy = df[b_covxy].to_numpy(dtype=np.float64)
    covyy = df[b_covyy].to_numpy(dtype=np.float64)

    n = len(pt1)
    if b_type1 and b_type2:
        type1 = df[b_type1].to_numpy(dtype=np.int32)
        type2 = df[b_type2].to_numpy(dtype=np.int32)
    else:
        # Fallback: assume tt (had-had) if not provided
        type1 = np.full(n, 1, dtype=np.int32)
        type2 = np.full(n, 1, dtype=np.int32)

    dm1 = df[b_dm1].to_numpy(dtype=np.int32) if b_dm1 else np.full(n, -1, dtype=np.int32)
    dm2 = df[b_dm2].to_numpy(dtype=np.int32) if b_dm2 else np.full(n, -1, dtype=np.int32)

    dm1 = np.where(type1 == 1, dm1, -1).astype(np.int32)
    dm2 = np.where(type2 == 1, dm2, -1).astype(np.int32)

    measuredTauLeptons = np.stack(
        [
            np.stack([type1, pt1, eta1, phi1, m1, dm1], axis=1),
            np.stack([type2, pt2, eta2, phi2, m2, dm2], axis=1),
        ],
        axis=1,
    ).astype(np.float64)

    covMET = np.stack(
        [
            np.stack([covxx, covxy], axis=1),
            np.stack([covxy, covyy], axis=1),
        ],
        axis=1,
    ).astype(np.float64)

    metx = met * np.cos(metphi)
    mety = met * np.sin(metphi)

    return {
        "measuredTauLeptons": measuredTauLeptons,
        "measuredMETx": metx,
        "measuredMETy": mety,
        "covMET": covMET,
    }

def load_input_file(file_path, tree_name=None, branches=None):
    if file_path.endswith(".root"):
        if tree_name is None or branches is None:
            raise ValueError("ROOT file should have tree_name and branches")
        return load_root_events(file_path, tree_name, branches)

    elif file_path.endswith(".parquet"):
        return load_events_parquet(file_path)

    elif file_path.endswith(".csv"):
        return load_events_csv(file_path)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")
