import numpy as np

class ControlledLengthModel():
    def __init__(self):

        self._T = 298 #°K
        self._kx = 0.068 #spring stiffness pN/nm
        self._xt = 10 #nm
        self._xm = 18 #nm

        self.k0 = 9.3e17 #1/s
        self.DeltaG0 = 264.0 #pN.nm
        self.DeltaG0t = 300 #pN.nm

        kB = 1.380649*1e-2 #pN.nm/K
        self._beta = 1/(kB*self._T)

        self.Deltaf = self._kx * self._xm
        self.xstar = self._xm - self._xt
        self.mu = (self._xt - self.xstar)/self._xm
        self.Omega = np.exp(-self._beta * (1 - self.mu**2)/8 * self._xm * self.Deltaf)
    
        self.xmax = 600 #nm
        self.xmin = 1e-2 #nm

    @property
    def xm(self):
        return self._xm
    @xm.setter
    def xm(self, val):
        self._xm = val
        self.xstar = val - self._xt
        self.mu = (self._xt - self.xstar)/val
        self.Deltaf = self.kx * val
        self.Omega = np.exp(-self._beta * (1 - self.mu**2)/8 * val * self.Deltaf)
    
    @property
    def kx(self):
        return self._kx
    @kx.setter
    def kx(self, val):
        self._kx = val
        self.Deltaf = val * self._xm
        self.Omega = np.exp(-self._beta * (1 - self.mu**2)/8 * self._xm * self.Deltaf)
    
    @property
    def xt(self):
        return self._xt
    @xt.setter
    def xt(self, val):
        self._xt = val
        self.xstar = self._xm - val
        self.mu = (val - self.xstar)/self._xm
    
    @property
    def T(self):
        return self._T
    @T.setter
    def T(self, val):
        self._T = val
        kB = 1.380649*1e-2 #pN.nm/K
        self._beta = 1/(kB*val)

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, val):
        self._beta = val
        kB = 1.380649*1e-2 #pN.nm/K
        self._T = 1/(kB*val)

    def configuration_energy(self, state:np.ndarray, length:np.ndarray)->np.ndarray:
        return 1/2 * self.kx * (length - self.xm*state)**2 + state*self.DeltaG0

    def force(self, state:np.ndarray, length:np.ndarray)->np.ndarray:
        force = np.zeros_like(length)
        if np.isscalar(length):
            if state:
                force = self.kx*(length-self.xm)
            else:
                force = self.kx*length
        else:
            force[state] = self.kx*(length[state]-self.xm)
            force[~state] = self.kx*length[~state]
        return force
    
    def length(self, state:np.ndarray, force:np.ndarray)->np.ndarray:
        length = np.zeros_like(force)
        if np.isscalar(force):
            if state:
                length = self.xm + force/self.kx
            else:
                length = force/self.kx
        else:
            length[state] = self.xm + force[state]/self.kx
            length[~state] = force[~state]/self.kx
        return length

    def f_(self, length:np.ndarray)->np.ndarray:
        return 1/2 * self.kx * (2*length - self.xm)

    def kFtoU(self, length:np.ndarray)->np.ndarray:
        return 1/self.Omega * self.k0 * np.exp(- self.beta * (self.DeltaG0t - self.f_(length) * self.xt))
    
    def kUtoF(self, length:np.ndarray)->np.ndarray:
        return 1/self.Omega * self.k0 * np.exp( - self.beta * (self.DeltaG0t - self.DeltaG0 + self.f_(length)*self.xstar))