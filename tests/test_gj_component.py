"""Tests for the Gottfried-Jackson (GJ) likelihood component.

Run with:  python -m pytest tests   (or simply: python tests/test_gj_component.py)
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

import FastMTT  # noqa: E402
import Likelihood  # noqa: E402
from Likelihood import gj_angle_from_x, gj_angle_max, gj_angle_measured  # noqa: E402

M_TAU = Likelihood.Likelihood().mTau
RNG = np.random.default_rng(12345)


def explicit_decay(p_tau, m_vis, cos_star):
    """Tau along z with momentum p_tau decays to a visible system of mass m_vis emitted at
    rest-frame polar angle acos(cos_star) and a massless neutrino.
    Returns visible (E, |p|), the lab angle between tau and visible system, and x = E_vis/E_tau."""
    E_tau = np.sqrt(p_tau**2 + M_TAU**2)
    beta, gamma = p_tau/E_tau, E_tau/M_TAU
    p_star = (M_TAU**2 - m_vis**2)/(2*M_TAU)
    E_star = (M_TAU**2 + m_vis**2)/(2*M_TAU)
    sin_star = np.sqrt(1 - cos_star**2)
    pz = gamma*(p_star*cos_star + beta*E_star)
    pt = p_star*sin_star
    E = gamma*(E_star + beta*p_star*cos_star)
    return E, np.hypot(pz, pt), np.arctan2(pt, pz), E/E_tau


def test_predicted_angle_matches_explicit_decay():
    p_tau = RNG.uniform(5, 300, 2000)
    m_vis = RNG.uniform(0.14, 1.6, 2000)
    cos_star = RNG.uniform(-1, 1, 2000)
    E_vis, p_vis, theta_true, x = explicit_decay(p_tau, m_vis, cos_star)
    theta = gj_angle_from_x(E_vis, p_vis, m_vis**2, x, M_TAU)
    np.testing.assert_allclose(theta, theta_true, rtol=1e-6, atol=1e-9)


def test_theta_max_is_maximum_over_x():
    p_vis = np.array([10.0, 40.0, 150.0])
    m_vis = np.array([0.14, 0.8, 1.3])
    E_vis = np.hypot(p_vis, m_vis)
    x = np.linspace(1e-4, 1, 200001)[:, None]
    theta = gj_angle_from_x(E_vis, p_vis, m_vis**2, x, M_TAU)
    np.testing.assert_allclose(theta.max(axis=0), gj_angle_max(p_vis, m_vis**2, M_TAU), rtol=1e-4)


def test_forbidden_x_maps_to_zero_angle():
    # x = 1 means a zero-energy neutrino, impossible for m_vis < m_tau
    assert gj_angle_from_x(50.0, np.sqrt(50.0**2 - 1.0), 1.0, 1.0, M_TAU) == 0.0


def test_measured_gradient_matches_finite_differences():
    PV = RNG.normal(0, 0.01, (50, 3))
    vis_dir = RNG.normal(size=(50, 3))
    vis_dir /= np.linalg.norm(vis_dir, axis=1)[:, None]
    SV = PV + 0.3*vis_dir + RNG.normal(0, 0.005, (50, 3))
    theta, grad_SV, L = gj_angle_measured(PV, SV, vis_dir)
    eps = 1e-7
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        num_SV = (gj_angle_measured(PV, SV + d, vis_dir)[0] - gj_angle_measured(PV, SV - d, vis_dir)[0])/(2*eps)
        num_PV = (gj_angle_measured(PV + d, SV, vis_dir)[0] - gj_angle_measured(PV - d, SV, vis_dir)[0])/(2*eps)
        np.testing.assert_allclose(num_SV, grad_SV[:, k], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(num_PV, -grad_SV[:, k], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(L, np.linalg.norm(SV - PV, axis=1))


def _events_with_known_gj(n=20, flight=0.3):
    """Hadronic legs along random directions with an SV placed exactly on the true tau line."""
    lik = Likelihood.Likelihood()
    p_tau = RNG.uniform(30, 120, (n, 2))
    m_vis = RNG.uniform(0.6, 1.4, (n, 2))
    cos_star = RNG.uniform(-0.9, 0.9, (n, 2))
    E_vis, p_vis, theta_true, x_true = explicit_decay(p_tau, m_vis, cos_star)
    # visible direction: random; tau direction: rotated by theta_true around a random perpendicular axis
    vis_dir = RNG.normal(size=(n, 2, 3))
    vis_dir /= np.linalg.norm(vis_dir, axis=-1)[..., None]
    perp = np.cross(vis_dir, RNG.normal(size=(n, 2, 3)))
    perp /= np.linalg.norm(perp, axis=-1)[..., None]
    tau_dir = np.cos(theta_true)[..., None]*vis_dir + np.sin(theta_true)[..., None]*perp
    PV = RNG.normal(0, 0.01, (n, 3))
    SV = PV[:, None, :] + flight*tau_dir
    P4 = np.concatenate([vis_dir*p_vis[..., None], E_vis[..., None]], axis=-1)
    lik.mvisleg1, lik.mvisleg2 = m_vis[:, 0].copy(), m_vis[:, 1].copy()   # normally set by FastMTT.run
    lik.setLeptonInputs(P4[:, 0], P4[:, 1], np.ones(n), np.ones(n), np.full(n, 10), np.full(n, 10))
    return lik, PV, SV, theta_true, x_true


def test_likelihood_peaks_where_predicted_angle_matches():
    lik, PV, SV, theta_true, x_true = _events_with_known_gj()
    lik.gj_model = "gauss"
    lik.gj_sigma_angle = 1e-5
    tiny = np.broadcast_to(np.eye(3)*1e-12, (len(PV), 3, 3))
    lik.setVertexInputs(PV, SV, tiny, np.broadcast_to(tiny[:, None], (len(PV), 2, 3, 3)), np.ones((len(PV), 2), bool))
    assert lik.gj_active.all()
    np.testing.assert_allclose(lik.gj_theta_meas, theta_true, rtol=1e-6)
    # Evaluate on a grid that contains the true (x1, x2) of every event
    grid = np.stack([x_true[:, 0], x_true[:, 1]], axis=1)
    f = lik.gjLikelihood(grid)
    np.testing.assert_allclose(np.diag(f), 1.0, rtol=1e-6)


def test_shared_pv_correlates_legs():
    lik, PV, SV, _, _ = _events_with_known_gj(n=5)
    n = len(PV)
    covSV = np.broadcast_to(np.eye(3)*1e-6, (n, 2, 3, 3))
    has = np.ones((n, 2), bool)
    lik.setVertexInputs(PV, SV, np.zeros((n, 3, 3)), covSV, has)
    assert np.allclose(lik.gj_inv_cov[:, 0, 1], 0.0)
    lik.setVertexInputs(PV, SV, np.broadcast_to(np.eye(3)*1e-6, (n, 3, 3)), covSV, has)
    assert not np.allclose(lik.gj_inv_cov[:, 0, 1], 0.0)
    # a leg without SV is decoupled
    has[:, 1] = False
    lik.setVertexInputs(PV, SV, np.broadcast_to(np.eye(3)*1e-6, (n, 3, 3)), covSV, has)
    assert np.allclose(lik.gj_inv_cov[:, 0, 1], 0.0) and not lik.gj_active[:, 1].any()


def test_sv_behind_pv_is_dropped():
    lik, PV, SV, _, _ = _events_with_known_gj(n=5)
    SV_flipped = 2*PV[:, None, :] - SV
    n = len(PV)
    cov = np.broadcast_to(np.eye(3)*1e-8, (n, 3, 3))
    lik.setVertexInputs(PV, SV_flipped, cov, np.broadcast_to(cov[:, None], (n, 2, 3, 3)), np.ones((n, 2), bool))
    assert not lik.gj_active.any()


def test_rice_model_removes_bias_of_gauss_model():
    """Measured angle = |true 2D offset + isotropic noise| with sigma = true angle.
    The best-fit predicted angle should be unbiased for 'rice' and too large for 'gauss'."""
    n, sigma, theta_true = 4000, 4e-3, 4e-3
    phi = RNG.uniform(0, 2*np.pi, n)
    off = theta_true*np.stack([np.cos(phi), np.sin(phi)], 1) + RNG.normal(0, sigma, (n, 2))
    theta_meas = np.linalg.norm(off, axis=1)
    lik = Likelihood.Likelihood()
    lik.gj_active = np.stack([np.ones(n, bool), np.zeros(n, bool)], 1)
    lik.gj_theta_meas = np.stack([theta_meas, np.zeros(n)], 1)
    lik.gj_sigma = np.stack([np.full(n, sigma), np.ones(n)], 1)
    lik.gj_inv_cov = np.broadcast_to(np.diag([1/sigma**2, 1.0]), (n, 2, 2))
    # bypass the kinematics: a 1D scan of the predicted angle itself
    grid = np.linspace(0, 0.02, 401)
    Likelihood_gj_angle_from_x = Likelihood.gj_angle_from_x
    try:
        Likelihood.gj_angle_from_x = lambda E, p, m2, x, mt: x*np.ones_like(E)
        lik.gj_E_vis = lik.gj_p_vis = lik.gj_m_vis2 = np.ones((n, 2))
        best = {}
        for model in ("gauss", "rice"):
            lik.gj_model = model
            f = lik.gjLikelihood(np.stack([grid, grid], 1))
            best[model] = grid[np.argmax(f, axis=1)]
    finally:
        Likelihood.gj_angle_from_x = Likelihood_gj_angle_from_x
    assert abs(np.mean(best["rice"]) - theta_true) < 0.15*theta_true
    assert np.mean(best["gauss"]) - theta_true > 0.3*theta_true


def _toy_inputs(n=40):
    leptons = np.zeros((n, 2, 6))
    leptons[:, :, 0] = 1                       # hadronic
    leptons[:, :, 1] = RNG.uniform(25, 80, (n, 2))
    leptons[:, :, 2] = RNG.uniform(-2, 2, (n, 2))
    leptons[:, 0, 3] = RNG.uniform(-np.pi, np.pi, n)
    leptons[:, 1, 3] = leptons[:, 0, 3] + np.pi + RNG.normal(0, 0.3, n)
    leptons[:, :, 4] = RNG.uniform(0.6, 1.3, (n, 2))
    leptons[:, :, 5] = 10
    metx, mety = RNG.normal(0, 20, n), RNG.normal(0, 20, n)
    covMET = np.broadcast_to(np.diag([300.0, 300.0]), (n, 2, 2)).copy()
    return leptons, metx, mety, covMET


def test_run_unchanged_without_usable_vertices():
    leptons, metx, mety, covMET = _toy_inputs()
    n = len(metx)
    base = FastMTT.FastMTT()
    base.myLikelihood.enableLikelihoodComponents(window=True)
    base.run(leptons.copy(), metx, mety, covMET)

    gj = FastMTT.FastMTT()
    gj.myLikelihood.enableLikelihoodComponents(window=True, GJ=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gj.run(leptons.copy(), metx, mety, covMET)            # no vertex inputs at all
    np.testing.assert_array_equal(gj.tau1pt, base.tau1pt)

    PV = np.zeros((n, 3))
    SV = np.zeros((n, 2, 3))
    cov = np.broadcast_to(np.eye(3)*1e-6, (n, 3, 3))
    gj.run(leptons.copy(), metx, mety, covMET, PV=PV, SV=SV, covPV=cov,
           covSV=np.broadcast_to(cov[:, None], (n, 2, 3, 3)))  # SV all zeros -> no leg used
    assert not gj.gj_used.any()
    np.testing.assert_array_equal(gj.tau1pt, base.tau1pt)
    np.testing.assert_array_equal(gj.tau2pt, base.tau2pt)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
