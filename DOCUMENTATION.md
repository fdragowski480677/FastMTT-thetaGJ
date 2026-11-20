# Documentation of FastMTT

## 1. Overview

Author: Wiktor Matyszkiewicz, Artur Kalinowski
Last update: 11.09.2025

Following repository contains a python implementation of the FastMTT algorithm, used to reconstruct invariant mass of di-tau system at high speed (around 1s for 1000 events on modern computer). For C++ version refer to: https://github.com/SVfit/ClassicSVfit/tree/fastMTT_2024.

Project structure:  
- **FastMTT.py** – high-level implementation of the reconstruction algorithm (grid scan, maximum likelihood selection, reconstruction of tau four-vectors).  
- **Likelihood.py** – implementation of the likelihood function, with different components: MET, tau-mass constraint, optionals.

### Installation

Clone the repository:
```
bash
git clone https://github.com/<your-username>/FastMTT.git
cd FastMTT
```

No special installation is required apart from standard Python libraries (see `requirements.txt`).

### Usage

Basic usage in Python:
```
import FastMTT

fMTT = FastMTT.FastMTT()
fMTT.run(measuredTauLeptons, measuredMETx, measuredMETy, covMET)

masses = fMTT.mass
```

Input and output should be contained in numpy arrays with the structure (N, ...), where N is the number of events. For inputs one should need:

1) measuredTauLeptons -- (N, 2, 6), with the structure (number_of_events, taon for reconstruction, kinematic parameters)

Kinematic parameters should be:

lepton[0]: decay_type:
1 - TauToHad
2 - TauToElec
3 - TauToMu

lepton[1]: pt
lepton[2]: eta
lepton[3]: phi
lepton[4]: mass
lepton[5]: hadron decay mode (-1 for non-hadrons)

2) measuredMETx -- (N,) array with the information of x component of reconstructed MET.
3) measuredMETy -- (N,) array with the information of y component of reconstructed MET.
4) covMET -- (N, 2, 2) array with the covariance matrix, containting information of transfer function (Gaussian) between true and reconstructed MET. Event-by-event information will work at best.

For the output one will obtain one array of the size (N,), containing estimated invariant masses.

### Quick Start

Minimal examples are contained in the batch_processing.py and visualisation.py:

```
python3 examples/batch_processing.py examples/example_data.csv
python3 examples/visualisation.py examples/example_data.csv
```

visualisation.py will produce example histograms of reconstructed mass and pt resolution, that will be stored in the ./images directory.

batch_processing.py will calculate batch of events using multiprocessing regime.

---

## 2. FastMTT.py

### Class **FastMTT**

The main class of the algorithm, responsible for:  
- configuration,  
- running the reconstruction,  
- grid scanning and likelihood evaluation,  
- selecting the best solution,  
- reconstructing tau four-vectors and their invariant mass,  
- optionally computing uncertainties and plotting the likelihood.  

#### Attributes:
- `myLikelihood` – instance of the `Likelihood` class.  
- `BestLikelihood` – best likelihood value.  
- `BestX` – optimal parameters (x1, x2).  
- `tau1P4`, `tau2P4` – reconstructed tau four-vectors.  
- `tau1pt`, `tau2pt` – reconstructed tau transvers momenta.
- `mass` – invariant mass of the ττ system.  
- `pt` – transverse momentum of the ττ system.  
- `PrintTime` - flag enabling measurement of execution time. False by default.
- `CalculateUncertainties` – flag enabling 1σ uncertainty computation.
- `min_masses`, `max_masses` – minimal and maximal mass within the assumed confidence limit (in case of uncertainty computation).
- `one_sigma` – one sigma interval calculated if `CalculateUncertainties` equals True.
- `WhichLikelihoodPlot` – event index for which a likelihood plot is saved. For -1 code does not produce any plot (default option).

#### Methods:

- **run(measuredTauLeptons, measuredMETx, measuredMETy, covMET)**  
  Main entry point of the algorithm for a given event.  
  - Checks that exactly two leptons are provided.  
  - Computes lepton four-vectors.  
  - Modifies lepton masses (depending on decay mode).  
  - Sets inputs for the `Likelihood`.  
  - Runs grid scan (`scan`).  
  - Stores best-fit tau four-vectors and reconstructed ττ mass.  
  - Optionally records execution time.    

- **scan()**  
  - Creates a parameter grid (x1, x2).  
  - Computes likelihood values over the grid.  
  - Picks the best solution.  
  - Optionally plots likelihood or computes uncertainties.  

- **plot_likelihood (event_number=0, threshold=None, filepath = None)**  
  Generate and save 2D plots of the likelihood function for selected events. By default it saves images into images/fastMTT/ directory. Called in the `scan` method.

- **get_p4(lepton)**  
  Converts lepton input (pt, η, φ, m) into a four-vector. Helper function. 

- **modify_lepton_mass(aLepton1)**  
  Adjusts lepton mass depending on decay type (electron, muon, hadronic). Helper function.

- **evaluate_mass(x)**
  Computes invariant mass for given fraction grid x.

- **contour_uncertainties(X1, X2, chi_square=2.3)**
Estimates mass uncertainties by evaluating the likelihood contour at 1σ (Δχ² = 2.3) by default.

Produces min_masses, max_masses, and one_sigma width.

---

### Helper functions

- **InvariantMass(p4)**  
  Computes the invariant mass of a four-vector or an array of four-vectors.  
  - **Input**: `p4` (array shape (...,4)).  
  - **Output**: `m` – invariant mass.  

- **pT(aP4)**  
  Computes the transverse momentum of a four-vector.  
  - **Input**: `aP4` (array shape (...,4)).  
  - **Output**: `pt` – transverse momentum.  

---

## 3. Likelihood.py

### Class **Likelihood**

Implements the likelihood function used for ττ system reconstruction.  
It uses inputs provided by `FastMTT` and evaluates the combined likelihood.

The main workflow starts with the `value` method, that calculates likelihood values for the points on the x1, x2 grid. Then, it refers to the proper likelihood components, that are multiplied in the calculation.
Deafult components are `massLikelihood()` and `metTF()`, while `Window()` or `mass_constraint()` are optional. One can manage them with the `enableLikelihoodComponents()` method. 

#### Attributes

- **MET inputs**  
  - `recoMET` (`np.ndarray`): Reconstructed MET 4-vector.  
  - `covMET` (`np.ndarray`): 2×2 covariance matrix of MET.

- **Parameters**  
  - `coeff1` (`float`): Power-law exponent in mass likelihood.  
  - `coeff2` (`float`): Scaling factor applied to mass.  

- **Lepton inputs**  
  - `leg1P4`, `leg2P4` (`np.ndarray`): 4-momenta of the two tau decay products.  
  - `mvis` (`np.ndarray`): Visible invariant mass of both leptons.  
  - `mvisleg1`, `mvisleg2` (`np.ndarray`): Visible invariant masses of individual leptons.  
  - `mVisOverTauSquare1`, `mVisOverTauSquare2` (`np.ndarray`): Squared ratios of visible mass to tau mass.  
  - `mTau` (`float`): Tau mass (in GeV), taken from physical constants.  
  - `leg1DecayType`, `leg2DecayType` (`np.ndarray`): Tau decay types (hadronic/leptonic).  
  - `leg1DecayMode`, `leg2DecayMode` (`np.ndarray`): Tau decay modes.  

- **Likelihood switches**  
  - `enable_MET` (`bool`): Enables MET likelihood component.  
  - `enable_mass` (`bool`): Enables mass likelihood component.  
  - `enable_mass_constraint` (`bool`): Enables Gaussian constraint around a target mass.  
  - `enable_window` (`bool`): Enables strict invariant mass window cut.  

- **Mass constraint parameters**  
  - `constraint_mean` (`float`): Target mass (e.g. Higgs 125 GeV, Z 91.2 GeV).  
  - `constraint_sigma` (`float`): Width of Gaussian constraint (default: 10 GeV).  
  - `window` (`list[float]`): Mass window boundaries (default: `[123, 127]`).  

---

### Methods

#### `setMassConstraint(mean, sigma)`
Set parameters for the Gaussian mass constraint.  

#### `setWindow(window)`
Set invariant mass window boundaries.  

#### `enableLikelihoodComponents(MET=None, mass=None, mass_constraint=None, window=None)`
Enable or disable individual likelihood components (booleans).  

#### `setLeptonInputs(aLeg1P4, aLeg2P4, aLeg1DecayType, aLeg2DecayType, aLeg1DecayMode, aLeg2DecayMode)`
Set lepton kinematics and decay information. Updates internal masses and ratios.  

#### `massLikelihood(m)`
Computes the mass-based likelihood for a given test mass `m`.  

#### `mass_constraint(invariant_mass)`
Applies Gaussian constraint centered around `constraint_mean`.  

#### `Window(invariant_mass)`
Applies a strict cut: returns mask of events within the mass window. 

#### `metTF(metP4, nuP4, covMET)`
Computes MET transfer function given reconstructed MET, neutrino system, and MET covariance.  

#### `value(x)`
Computes the full likelihood value for a given parameter grid `x`.  
Includes contributions from MET, mass, and optional constraints.

---

## 4. FastMTT_IO.py

This module provides helper methods to run the [FastMTT](https://github.com/cms-tau-pog/FastMTT) algorithm in parallel
and to load event data from **ROOT** or **CSV** files into the required format.

It combines:
- **Processing functions**: parallel execution of FastMTT reconstruction.
- **I/O functions**: loading events from different file formats and preparing them for FastMTT.

Parallel execution is recommended, especially for the number of events > 10 thousands (memory allocation is very big in these cases without batching).

### FastMTT Processing

#### `init_worker()`
Initializes a global `FastMTT` instance for each worker process.
This avoids repeatedly constructing the object inside the processing loop.

#### `process_batches_for_worker(args)`
Processes a list of event batches assigned to a worker.
`args` *(tuple)*: `(worker_id, worker_batches)`

#### `process_FastMTT(measuredTauLeptons, xMETs, yMETs, covMETs, batch_size=5000, num_workers=4)`
Runs the FastMTT algorithm in parallel on the provided events.

**Returns**
- `mFast` *(np.ndarray)*: Higgs candidate invariant masses.
- `ptFast` *(np.ndarray)*: Higgs candidate transverse momenta.
- `tau1pt` *(np.ndarray)*: Reconstructed `pT` of the first tau.
- `tau2pt` *(np.ndarray)*: Reconstructed `pT` of the second tau.

---

### Data Loading

#### `read_root_file(file_path, tree_name, branches, entry_stop=None)`
Reads a ROOT file and extracts branches; used in `load_root_events(...)`

#### `load_root_events(file_path, tree_name, branches, entry_stop=None)`
Loads tau candidates and MET information from a ROOT file and prepares it for FastMTT.

#### `load_events_csv(csv_data)`
Loads tau candidates and MET information from a CSV file.

#### `load_input_file(file_path, tree_name=None, branches=None)`
Generic loader that automatically chooses the correct loading function
depending on the file extension.
---

## 5. Gottfried-Jackson (GJ) angle component

An optional likelihood component that uses the tau flight direction measured from the
primary vertex (PV) to the refitted secondary vertex (SV). It is disabled by default.

```python
fMTT = FastMTT.FastMTT()
fMTT.myLikelihood.enableLikelihoodComponents(MET=True, mass=True, mass_constraint=False, window=True, GJ=True)
fMTT.run(measuredTauLeptons, measuredMETx, measuredMETy, covMET,
         PV=PV, SV=SV, covPV=covPV, covSV=covSV, hasSV=hasSV)
fMTT.gj_used   # (N, 2) bool, legs that entered the component
```

Extra inputs (positions in cm, covariances in cm²):
- `PV` (N, 3), `SV` (N, 2, 3), `covPV` (N, 3, 3), `covSV` (N, 2, 3, 3)
- `hasSV` (N, 2) bool. If omitted, a leg has an SV when its SV position is not all zeros.

From the command line:

```
python examples/batch_processing.py file.parquet --channel tt --window --gj
```

### Method

The GJ angle θ is the opening angle between the tau and its visible decay products.

1. **Measured angle.** n = (SV − PV)/|SV − PV| is the flight direction and v the direction of the
   reconstructed visible momentum, so cos θ_meas = n·v.
2. **Predicted angle.** For an energy fraction x = E_vis/E_tau, a massless invisible system
   (one neutrino, hadronic decays) requires
   `cos θ(x) = (2 E_tau E_vis − m_tau² − m_vis²) / (2 |p_tau| |p_vis|)` with E_tau = E_vis/x.
   Kinematically forbidden x give cos θ > 1 and are mapped to θ = 0.
   The largest allowed angle is sin θ_max = (m_tau² − m_vis²)/(2 m_tau |p_vis|).
3. **Error propagation (analytic).** With w = v − cos θ n and the unit vector u = w/sin θ,
   ∂θ/∂SV = −u/L and ∂θ/∂PV = +u/L, where L is the flight length. Hence
   σ²(θ_i) = (u_iᵀ (C_SV,i + C_PV) u_i)/L_i² + σ_angle², and the two legs are correlated through
   the shared PV: cov(θ_1, θ_2) = u_1ᵀ C_PV u_2 /(L_1 L_2). Only the error perpendicular to the
   flight direction enters, so the large SV uncertainty along the flight line does not matter.
   σ_angle (`gj_sigma_angle`, default 1.7 mrad) covers the visible-direction resolution and
   residual effects.
4. **Likelihood.** `gj_model="gauss"` (default): the bivariate normal distribution of
   r = (θ_meas,1 − θ(x_1), θ_meas,2 − θ(x_2)) with the covariance above. Legs without SV are
   decoupled and contribute a constant. `gj_model="rice"`: each leg's 2D direction offset is taken
   as bivariate normal and the unknown azimuth of the tau around the visible direction is
   integrated out, which multiplies the Gaussian term by `i0e(θ_meas θ(x)/σ²)`.

A leg enters the component only if it is hadronic, has an SV, and
(θ_meas − θ_max)/σ < `gj_max_pull` (default 5). The cut removes SVs reconstructed behind the PV.
Per-event constant factors are dropped because they do not move the maximum.

### Vertex covariance placeholder

The Run3_2022 ntuples contain PV and SV positions but no vertex covariances.
`FastMTT_IO.placeholder_vertex_covariances` provides a constant model calibrated on these files:
the SV displacement perpendicular to the generated tau direction, fitted versus flight length as
σ²(L) = s_v² + (s_a L)², gives s_v ≈ 10 µm transverse and ≈ 17 µm along z with the
beamspot-constrained PV. The split between PV and SV is an assumption. Pulls of θ_meas against
the generated GJ angle have a width of about 1.1 on average, but the model underestimates the
error of the best-measured legs (width 1.2–1.5 for σ/θ_max < 0.5). Replace it with event-by-event
covariances once they are stored.

Refitted SVs exist only for 3-prong decays (decay modes 10, 11), about 30% of hadronic legs.

### Options

| attribute / CLI flag | default | meaning |
|---|---|---|
| `enable_GJ` / `--gj` | off | enable the component |
| `--pv` | `PVBS` | PV columns: beamspot-constrained `PVBS_*` or `PV_*` |
| `gj_model` / `--gj-model` | `gauss` | likelihood shape, `gauss` or `rice` |
| `gj_sigma_angle` / `--gj-sigma-angle` | 1.7e-3 | constant angular resolution [rad] |
| `gj_max_pull` / `--gj-max-pull` | 5 | quality cut on the measured angle |
| `--gj-cov-scale` | 1 | scale on the placeholder vertex resolution |

`batch_processing.py` also gained `--output`, `--plot-dir` and the `et` channel.

### Validation

`examples/compare_resolution.py` compares outputs of `batch_processing.py`. Resolution is
r = (pT_reco − pT_gen)/pT_gen for hadronic legs only (|genPart_pdgId| = 15), because for electron
and muon legs `genPart_pt` is the visible lepton.

Tests: `python -m pytest tests`.

Results on Run3_2022 ggH→ττ, window on, 68% half-width of r (lower is better):

| channel | legs | baseline | + GJ gauss | + GJ rice |
|---|---|---|---|---|
| tt | 3-prong with SV | 0.157 | 0.148 | 0.157 |
| mt | 3-prong with SV | 0.162 | 0.150 | 0.158 |
| et | 3-prong with SV | 0.163 | 0.151 | 0.160 |
| tt | all hadronic | 0.161 | 0.156 | 0.160 |
| mt | all hadronic | 0.171 | 0.167 | 0.170 |
| et | all hadronic | 0.174 | 0.170 | 0.172 |

The fraction of 3-prong legs within 10% of the generated pT rises from about 0.46 to 0.49–0.50
with `gauss`. Results change by less than 0.002 when the placeholder vertex resolution is scaled
by 0.7 or 1.5, when `gj_sigma_angle` is 1.0 or 2.5 mrad, with `PV_*` instead of `PVBS_*`, or with
`gj_max_pull` = 3. The parameters were checked on tt only; mt and et are independent.

Interpretation: `examples/gj_diagnostics.py` splits the gain by angle precision σ/θ_max and runs a
null test. In the null test the measured angles, in units of θ_max, are shuffled between legs of
similar σ/θ_max and FastMTT is rerun. The angle then carries no information about its leg, while the
likelihood shapes stay the same. Change of the 68% half-width for 3-prong legs with SV, tt+mt+et
(statistical uncertainty ±0.4%):

| model | real angles | shuffled angles |
|---|---|---|
| `gauss` | −6.5% | −4.4% |
| `rice` | −1.7% | +0.1% |

About two thirds of the `gauss` gain therefore does not come from the vertex measurement. When
σ ≳ θ_max the measured angle is mostly noise and usually exceeds θ_max. `gauss` then favours the x
with the largest θ(x) and penalises x near the kinematic limits, where θ(x) → 0. It does so in the
same way for every such leg, which makes it a prior on x. `rice` integrates over the unknown azimuth
and becomes flat in x for these legs. The genuine information, about 2% and similar in both models,
sits in the legs with σ/θ_max < 0.5, where `rice` gives −4.2%. Event-by-event vertex covariances
should make the component more informative.

The component costs about 40–50% extra run time (tt: 17 s → 25 s on 8 workers).
