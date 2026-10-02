import numpy as np


class BEModel():
    """Two-state Bell-Evans model of a DNA hairpin.

    Kinetic rates in the force ensemble (LdF):
        k_F->U(f) = k0 * exp( beta * f * xt )
        k_U->F(f) = k0 * exp( beta * (DeltaG0 - f * xstar) ),   xstar = xm - xt

    k0 is the zero-force UNFOLDING rate. The attempt rate k_a and the barrier
    height DeltaG0t only ever enter through k0 = k_a * exp(-beta * DeltaG0t),
    so they are not parameters of the model.

    Default parameters: hairpin L4, Rico-Pasto et al., Phys. Rev. X 11, 031052
    (2021), Appendix C. They are identical in dynamics to the previous
    parametrization (k_a = 9.3e17 s^-1, DeltaG0t = 300 pN.nm).
    """

    def __init__(self, ensemble: str = 'LdF'):

        #### PHYSICAL CONSTANTS ####
        self.kB = 1.380649e-2  # [pN.nm/K] Boltzmann constant

        #### USER-DEFINED VARIABLES ####
        self._T = 298  # [K] temperature
        self._beta = 1/(self.kB*self._T)  # [pN^-1.nm^-1] coldness
        self._k0 = 2e-14  # [s^-1] zero-force unfolding rate k_F->U(0)
        self._xm = 18  # [nm] xU - xF
        self._DeltaG0 = 264.0  # [pN.nm] GU - GF at zero force
        self._xt = 9  # [nm] x(transition state) - xF
        self._kx = 0.068  # [pN/nm] spring stiffness of the optical trap

        #### CONSTRAINED VARIABLES ####
        self._update_constrained_variables()
        self.ensemble = ensemble  # LdF controlled force, FdL controlled length
        self._update_functions()

        #### CACHE FOR MEMOIZATION ####
        self._cached_results = {}

    def _update_constrained_variables(self):
        self.fc = self._DeltaG0 / self._xm  # [pN] coexistence force
        self.Deltaf = self._kx * self._xm
        self.xstar = self._xm - self._xt
        self.mu = (self._xt - self.xstar) / self._xm
        self.Omega = np.exp(-(self._beta * (1 - self.mu**2) / 8 * self._xm * self.Deltaf))

    @property
    def kc(self):
        """[s^-1] rate at coexistence, k_F->U(fc) = k_U->F(fc)"""
        return self._k0 * np.exp(self._beta * self.fc * self._xt)

    @property
    def T(self):
        return self._T
    @T.setter
    def T(self, val):
        self._T = val
        self._beta = 1/(self.kB*val)
        self._update_constrained_variables()

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, val):
        self._beta = val
        self._T = 1/(self.kB*val)
        self._update_constrained_variables()

    @property
    def k0(self):
        return self._k0
    @k0.setter
    def k0(self, val):
        self._k0 = val

    @property
    def xm(self):
        return self._xm
    @xm.setter
    def xm(self, val):
        self._xm = val
        self._update_constrained_variables()

    @property
    def DeltaG0(self):
        return self._DeltaG0
    @DeltaG0.setter
    def DeltaG0(self, val):
        self._DeltaG0 = val
        self._update_constrained_variables()

    @property
    def xt(self):
        return self._xt
    @xt.setter
    def xt(self, val):
        self._xt = val
        self._update_constrained_variables()

    @property
    def DeltaG0t(self):
        raise AttributeError(
            "DeltaG0t was removed: the model is parametrized by the zero-force "
            "rate k0 = k_a*exp(-beta*DeltaG0t). Use model.k0 instead.")
    @DeltaG0t.setter
    def DeltaG0t(self, val):
        raise AttributeError(
            "DeltaG0t was removed: set model.k0 = k_a*np.exp(-model.beta*DeltaG0t) instead.")

    @property
    def kx(self):
        return self._kx
    @kx.setter
    def kx(self, val):
        self._kx = val
        self._update_constrained_variables()

    @property
    def ensemble(self):
        return self._ensemble
    @ensemble.setter
    def ensemble(self, val):
        self._ensemble = val
        self._update_functions()

    def _update_functions(self):
        if self._ensemble == 'LdF':
            self.kFtoU = lambda force: self.k0 * np.exp(self.beta * force*self._xt)
            self.kUtoF = lambda force: self.k0 * np.exp(self.beta * (self._DeltaG0 - force*(self._xm - self._xt)))
            self.DeltaG = lambda state, force: state*(self._DeltaG0 - force * self._xm)
        elif self._ensemble == 'FdL':
            f_ = lambda length: 1/2 * self._kx * (2*length - self._xm)
            self.kFtoU = lambda force: 1/self.Omega * self.k0 * np.exp(self.beta * f_(force/self._kx)*self._xt)
            self.kUtoF = lambda force: 1/self.Omega * self.k0 * np.exp(self.beta * (self._DeltaG0 - f_(self._xm + force/self._kx)*self.xstar))
            self.DeltaG = lambda state, force: self._DeltaG0 + 1/2 * self._kx * (self.length(state, force) - self._xm*state)**2

    def PsF(self, f1: np.ndarray, f2: np.ndarray, r: np.ndarray) -> np.ndarray:
        """returns the survival probability of F from f1 to f2 at pull rate r in the forward process"""
        return np.exp(1 / (self.beta * self.xt * r) * (self.kFtoU(f1) - self.kFtoU(f2)))

    def PsU(self, f1: np.ndarray, f2: np.ndarray, r: np.ndarray) -> np.ndarray:
        """returns the survival probability of U from f1 to f2 at pull rate r in the forward process"""
        return np.exp(1 / (self.beta * (self.xm - self.xt) * r) * (self.kUtoF(f2) - self.kUtoF(f1)))

    def force(self, state: np.ndarray, length: np.ndarray) -> np.ndarray:
        force = np.zeros_like(length)
        if np.isscalar(length):
            force = self.kx*(length-self.xm) if state else self.kx*length
        else:
            force[state], force[~state] = self.kx*(length[state]-self.xm), self.kx*length[~state]
        return force

    def length(self, state: np.ndarray, force: np.ndarray) -> np.ndarray:
        length = np.zeros_like(force)
        if np.isscalar(force):
            length = self.xm + force/self.kx if state else force/self.kx
        else:
            length[state], length[~state] = self.xm + force[state]/self.kx, force[~state]/self.kx
        return length