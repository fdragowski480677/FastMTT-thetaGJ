import numpy as np
from scipy.constants import physical_constants
from scipy.special import i0e

def InvariantMass(p4):
    metric = np.array([-1,-1,-1,1])
    p4_square = p4*(metric*p4)
    m = np.sqrt(np.sum(p4_square, axis=-1))
    return m

###############################################################################
# Gottfried-Jackson (GJ) angle
#
# theta_GJ is the opening angle between the tau flight direction and the momentum
# of its visible decay products. It is obtained in two independent ways:
#
#  1) measured:  flight direction = PV -> SV vector, visible direction = reco p_vis
#  2) predicted: for an energy fraction x = E_vis / E_tau, requiring a massless
#                invisible system (one neutrino in hadronic decays) gives
#                cos(theta_GJ) = (2 E_tau E_vis - m_tau^2 - m_vis^2) / (2 |p_tau| |p_vis|)
#
# The GJ likelihood component compares 1) and 2) with a (multivariate) normal
# distribution whose covariance is propagated from the PV and SV covariances.
###############################################################################

def gj_angle_from_x(E_vis, p_vis, m_vis2, x, m_tau):
    """Predicted GJ angle [rad] for energy fraction x = E_vis / E_tau.

    All arguments broadcast against each other. Kinematically forbidden values
    (cos > 1, or E_tau < m_tau) are mapped to the closest allowed angle, 0.
    """
    E_tau = E_vis / x
    p_tau = np.sqrt(np.maximum(E_tau**2 - m_tau**2, 0.0))
    num = 2.0*E_tau*E_vis - m_tau**2 - m_vis2
    den = 2.0*p_tau*p_vis
    cos_gj = np.where(den > 0, num/np.where(den > 0, den, 1.0), 1.0)
    return np.arccos(np.clip(cos_gj, -1.0, 1.0))


def gj_angle_max(p_vis, m_vis2, m_tau):
    """Largest GJ angle allowed for a visible system of momentum p_vis and mass^2 m_vis2.

    sin(theta_max) = (m_tau^2 - m_vis^2) / (2 m_tau |p_vis|), obtained by maximising
    gj_angle_from_x over the tau energy at fixed visible momentum.
    """
    sin_max = (m_tau**2 - m_vis2) / (2.0*m_tau*np.where(p_vis > 0, p_vis, 1.0))
    return np.arcsin(np.clip(sin_max, 0.0, 1.0))


def gj_angle_measured(PV, SV, vis_dir):
    """Measured GJ angle and its analytic gradient.

    PV, SV, vis_dir: arrays (..., 3); vis_dir must be a unit vector.
    Returns theta (...), grad_SV (..., 3) = d theta / d SV, and the flight length L (...).
    The gradient w.r.t. the PV is -grad_SV.

    Derivation: n = (SV - PV)/L, cos(theta) = n.v. Moving the SV by d changes n by
    (d - (n.d) n)/L, hence d cos(theta) = w.d / L with w = v - cos(theta) n, |w| = sin(theta),
    and d theta = -u.d / L with the unit vector u = w / sin(theta) (perpendicular to n,
    pointing from the flight direction towards the visible direction).
    """
    F = SV - PV
    L = np.linalg.norm(F, axis=-1)
    L_safe = np.where(L > 0, L, 1.0)
    n = F / L_safe[..., None]
    cos_t = np.sum(n*vis_dir, axis=-1)
    w = vis_dir - cos_t[..., None]*n
    sin_t = np.linalg.norm(w, axis=-1)
    theta = np.arctan2(sin_t, cos_t)
    u = w / np.where(sin_t > 0, sin_t, 1.0)[..., None]
    grad_SV = -u / L_safe[..., None]
    return theta, grad_SV, L


class Likelihood:
    def __init__(self):
        #METinputs
        self.recoMET = np.array([0.0, 0.0, 0.0, 0.0])
        self.covMET = np.ones((2, 2))

        #setParameters
        self.coeff1 = 6
        self.coeff2 = 1/1.15

        #LeptonInputs
        self.leg1P4 = np.array([0.0, 0.0, 0.0, 0.0])
        self.leg2P4 = np.array([0.0, 0.0, 0.0, 0.0])

        #Visible mass of both leptons
        self.mvis = np.array([0.0, 0.0, 0.0, 0.0])

        #Invariant mass of each lepton
        self.mvisleg1 = np.array([0.0])
        self.mvisleg2 = np.array([0.0])

        self.mVisOverTauSquare1 = np.array([0.0])
        self.mVisOverTauSquare2 = np.array([0.0])
         
        self.mTau = physical_constants['tau energy equivalent'][0]/1000 #MeV -> GeV
        
        self.leg1DecayType = np.array([0.0])
        self.leg2DecayType = np.array([0.0])
        self.leg1DecayMode = np.array([0.0])
        self.leg2DecayMode = np.array([0.0])

        #Enable/disable likelihood channel
        self.enable_MET = True
        self.enable_mass = True

        #Mass constraints of two type -- strict cut (window) or additional likelihood component with normal distribution (mass_constraint)
        self.enable_mass_constraint = False
        self.constraint_mean = 125  # Mass of the particle, that we want to reconstruct. In this case, it is Higgs mass
        
        # For Z0:
        #self.constraint_mean = 91.1876
        self.constraint_sigma = 10 #artificially set value, one can play with it and adjust for the best mass/pt resolution
        #However something around 10GeV seems to work optimally

        self.enable_window = False
        self.window = [123, 127]

        #These are experimental and not used by main code
        self.enable_px = False
        self.enable_py = False

        #Gottfried-Jackson angle component from the PV -> SV flight direction (needs setVertexInputs)
        self.enable_GJ = False
        #Constant angular resolution [rad] added in quadrature to the vertex term:
        #visible direction resolution and residual effects (see placeholder calibration in FastMTT_IO.py)
        self.gj_sigma_angle = 1.7e-3
        #A leg is not used if its measured angle exceeds the largest physical one by more than gj_max_pull sigma
        #(e.g. SV reconstructed behind the PV)
        self.gj_max_pull = 5.0
        #"gauss": normal distribution of the polar angles of both legs, correlated through the shared PV
        #         (profile over the unknown azimuth of the tau around the visible direction)
        #"rice":  bivariate normal on each 2D direction offset, integrated over that azimuth (Rice distribution);
        #         statistically unbiased when sigma ~ theta but less constraining. On Run3_2022 tt the
        #         "gauss" model gave the better tau pT resolution, so it is the default.
        self.gj_model = "gauss"
        self.clearVertexInputs()

        return

    def setMassConstraint(self, mean, sigma):
        self.constraint_mean = mean
        self.constraint_sigma = sigma

    def setWindow(self, window):
        self.window = window

    def enableLikelihoodComponents(self, MET = None, mass = None, mass_constraint = None, window = None, phi2 = None, GJ = None):  #All Boolean
        if GJ is not None:
            self.enable_GJ = GJ
        if MET is not None:
            self.enable_MET = MET
        if mass is not None:
            self.enable_mass = mass
        if mass_constraint is not None:
            self.enable_mass_constraint = mass_constraint
        if window is not None:
            self.enable_window = window

    def setLeptonInputs(self, aLeg1P4, aLeg2P4, aLeg1DecayType, aLeg2DecayType, aLeg1DecayMode, aLeg2DecayMode):
        
        self.leg1DecayType = aLeg1DecayType
        self.leg2DecayType = aLeg2DecayType
        self.leg1DecayMode = aLeg1DecayMode
        self.leg2DecayMode = aLeg2DecayMode

        self.leg1P4 = aLeg1P4
        self.leg2P4 = aLeg2P4
        
        #visible invariant mass
        #eq. (4)
        self.mvis = InvariantMass(self.leg1P4 + self.leg2P4)
        
        self.mvisleg1[(aLeg1DecayType==1) & (self.mvisleg1>1.5)] = 0.3
        self.mvisleg2[(aLeg2DecayType==1) & (self.mvisleg2>1.5)] = 0.3

        self.mVisOverTauSquare1 = (self.mvisleg1/self.mTau)**2
        self.mVisOverTauSquare2 = (self.mvisleg2/self.mTau)**2

    def clearVertexInputs(self):
        """Forget vertex inputs; the GJ component then evaluates to 1."""
        self.gj_active = None       # (N, 2) legs entering the GJ component
        self.gj_theta_meas = None   # (N, 2) measured GJ angle [rad]
        self.gj_sigma = None        # (N, 2) its uncertainty [rad]
        self.gj_theta_max = None    # (N, 2) largest physical GJ angle [rad]
        self.gj_inv_cov = None      # (N, 2, 2) inverse covariance of (theta_1, theta_2)

    def setVertexInputs(self, PV, SV, covPV, covSV, hasSV):
        """Prepare the GJ component. Call after setLeptonInputs.

        PV: (N, 3) primary vertex, SV: (N, 2, 3) secondary vertex of each leg,
        covPV: (N, 3, 3), covSV: (N, 2, 3, 3) covariances (same length unit as the positions),
        hasSV: (N, 2) bool, True where the leg has a valid SV.

        The covariance of the measured angles is propagated linearly. The PV is shared by both
        legs, which makes theta_1 and theta_2 correlated: cov_12 = g1^T covPV g2.
        Only the error perpendicular to the flight direction matters, so a large SV uncertainty
        along the flight direction does not degrade the angle.
        """
        PV = np.asarray(PV, dtype=np.float64)
        SV = np.asarray(SV, dtype=np.float64)
        covPV = np.asarray(covPV, dtype=np.float64)
        covSV = np.asarray(covSV, dtype=np.float64)

        visP4 = np.stack([self.leg1P4, self.leg2P4], axis=1)   # (N, 2, 4)
        p_vec = visP4[..., :3]
        p_vis = np.linalg.norm(p_vec, axis=-1)
        vis_dir = p_vec / np.where(p_vis > 0, p_vis, 1.0)[..., None]

        theta, grad_SV, L = gj_angle_measured(PV[:, np.newaxis, :], SV, vis_dir)
        grad_PV = -grad_SV

        var = (np.einsum('nli,nlij,nlj->nl', grad_SV, covSV, grad_SV)
               + np.einsum('nli,nij,nlj->nl', grad_PV, covPV, grad_PV)
               + self.gj_sigma_angle**2)
        cov12 = np.einsum('ni,nij,nj->n', grad_PV[:, 0], covPV, grad_PV[:, 1])

        m_vis2 = np.clip(visP4[..., 3]**2 - p_vis**2, 0.0, (0.999*self.mTau)**2)
        theta_max = gj_angle_max(p_vis, m_vis2, self.mTau)

        # The massless-neutrino relation only holds for hadronic decays (decay type 1)
        hadronic = np.stack([self.leg1DecayType == 1, self.leg2DecayType == 1], axis=1)
        with np.errstate(invalid='ignore'):
            active = (np.asarray(hasSV, dtype=bool) & hadronic & (L > 0)
                      & np.isfinite(theta) & np.isfinite(var)
                      & (theta - theta_max < self.gj_max_pull*np.sqrt(var)))

        # Unused legs are decoupled: unit variance, zero correlation, zero residual
        var = np.where(active, var, 1.0)
        cov12 = np.where(active[:, 0] & active[:, 1], cov12, 0.0)
        det = var[:, 0]*var[:, 1] - cov12**2
        inv_cov = np.empty((len(det), 2, 2))
        inv_cov[:, 0, 0] = var[:, 1]/det
        inv_cov[:, 1, 1] = var[:, 0]/det
        inv_cov[:, 0, 1] = inv_cov[:, 1, 0] = -cov12/det

        self.gj_active = active
        self.gj_theta_meas = np.where(active, theta, 0.0)
        self.gj_sigma = np.sqrt(var)
        self.gj_theta_max = theta_max
        self.gj_inv_cov = inv_cov
        self.gj_E_vis = visP4[..., 3]
        self.gj_p_vis = p_vis
        self.gj_m_vis2 = m_vis2

    def gjLikelihood(self, x):
        """GJ component on the grid x (G, 2). Returns (N, G).

        gauss: exp(-chi2/2) with chi2 = r^T C^-1 r, r_i = theta_meas_i - theta_GJ(x_i).
               Comparing only the polar angle is the profile over the unknown azimuth of the
               tau around the visible direction. It is biased towards large angles when the
               resolution is comparable to the angle, because the measured angle is the length
               of a 2D offset and noise always makes it longer on average.
        rice:  the 2D direction offset is bivariate normal (per-leg sigma) and the azimuth is
               integrated out, giving exp(-(tm^2 + tp^2)/(2 s^2)) I0(tm tp / s^2) per leg
               = gauss term * i0e(tm tp / s^2). The gauss term keeps the PV correlation.

        Per-event constants (normalisation, chi2 offset) do not change the position of the
        maximum, so chi2 is shifted to 0 at the best grid point, which also protects against
        underflow.
        """
        if self.gj_active is None or not np.any(self.gj_active):
            return np.ones((len(self.mvis), x.shape[0]))

        theta_pred = gj_angle_from_x(self.gj_E_vis[:, np.newaxis, :], self.gj_p_vis[:, np.newaxis, :],
                                     self.gj_m_vis2[:, np.newaxis, :], x[np.newaxis, :, :], self.mTau)
        r = np.where(self.gj_active[:, np.newaxis, :], self.gj_theta_meas[:, np.newaxis, :] - theta_pred, 0.0)
        chi2 = np.einsum('ngi,nij,ngj->ng', r, self.gj_inv_cov, r)
        if self.gj_model == "rice":
            z = self.gj_theta_meas[:, np.newaxis, :]*theta_pred/self.gj_sigma[:, np.newaxis, :]**2
            chi2 -= 2.0*np.sum(np.where(self.gj_active[:, np.newaxis, :], np.log(i0e(z)), 0.0), axis=-1)
        elif self.gj_model != "gauss":
            raise ValueError(f"Unknown gj_model '{self.gj_model}', expected 'rice' or 'gauss'")
        chi2 -= chi2.min(axis=1, keepdims=True)
        return np.exp(-0.5*chi2)

    def massLikelihood(self, m):
        mScaled = m*self.coeff2

        mask1 = (mScaled < self.mvis[:, np.newaxis])
        
        mVS2 = (self.mvis[:, np.newaxis]/mScaled)**2
        
        x1Min = np.minimum(1.0, self.mVisOverTauSquare1)
        x2Min = np.maximum(self.mVisOverTauSquare2[:, np.newaxis], mVS2)
        x2Max = np.minimum(1.0, mVS2/x1Min[:, np.newaxis])
        
        mask2 = (x2Min > x2Max)
        
        jacobiFactor = 2.0*self.mvis[:, np.newaxis]**2*mScaled**(-self.coeff1)
        x2IntegralTerm = np.log(x2Max/x2Min)

        value = 0.0
        value += x2IntegralTerm

        HadDecay1 = np.broadcast_to((self.leg1DecayType != 1)[:, np.newaxis], value.shape)
        value += HadDecay1 * mVS2 * (1 / x2Max - 1 / x2Min)

        HadDecay2 = np.broadcast_to((self.leg2DecayType != 1)[:, np.newaxis], value.shape)
        value += HadDecay2 * (mVS2*x2IntegralTerm - (x2Max - x2Min))

        value[mask1 | mask2] = 0.0

        value *= 1E9*jacobiFactor

        return value

    def mass_constraint(self, invariant_mass):

        Gauss_factor = np.exp(-(invariant_mass - self.constraint_mean)**2/(2*self.constraint_sigma**2))

        return Gauss_factor

    def Window(self, invariant_mass):
        mask = (invariant_mass > self.window[0]) & (invariant_mass < self.window[1])
        return mask
    
    def metTF(self, metP4, nuP4, covMET):
        aMETx = metP4[..., 0]
        aMETy = metP4[..., 1]

        covDET = np.linalg.det(covMET)


        mask = covDET < 1E-10
        covDET[mask] = 1.0

        constMET = 1/2/np.pi/np.sqrt(covDET)
        residualX = aMETx[:, np.newaxis] - nuP4[:, :, 0]
        residualY = aMETy[:, np.newaxis] - nuP4[:, :, 1]

        #covMET 1 coordinate responds to X and 0 coordinate to Y
        pull2 = residualX*(covMET[:, np.newaxis, 1, 1]*residualX - covMET[:, np.newaxis, 0, 1]*residualY) + residualY*(-covMET[:, np.newaxis, 1, 0]*residualX + covMET[:, np.newaxis, 0, 0]*residualY)
        pull2 /= covDET[:, np.newaxis]
        
        pull2[np.broadcast_to(mask[:, np.newaxis], pull2.shape)] = 0.0
        return constMET[:, np.newaxis]*np.exp(-0.5*pull2)
    
    def value(self, x):
        
        x1Min = np.minimum(1.0, self.mVisOverTauSquare1)
        x2Min = np.minimum(1.0, self.mVisOverTauSquare2)

        mask = (x[:, 0] < x1Min[:, np.newaxis]) | (x[:, 1] < x2Min[:, np.newaxis])
        
        testP4 = self.leg1P4[:, np.newaxis, :] / x[:, 0][:, np.newaxis] + self.leg2P4[:, np.newaxis, :] / x[:, 1][:, np.newaxis]

        testMET = testP4 - self.leg1P4[:, np.newaxis, :] - self.leg2P4[:, np.newaxis, :]

        value = np.full(testMET.shape[:2], -1.0) #Negative likelihood

        if self.enable_MET:
            value *= self.metTF(self.recoMET, testMET, self.covMET)
        
        if self.enable_mass:
            value *= self.massLikelihood(InvariantMass(testP4))

        if self.enable_mass_constraint:
            value *= self.mass_constraint(InvariantMass(testP4))

        if self.enable_GJ:
            value *= self.gjLikelihood(x)
        
        if not self.enable_window: #default
            value[mask] = 0.000001
        else:
            value[~self.Window(InvariantMass(testP4))] = 1.0

        return value