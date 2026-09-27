import mpmath as mp
import numpy as np
from scipy.integrate import quad

class ControlledForceModel():
    def __init__(self):

        self._T = 298 #K

        self.k0 = 9.3e17  #1/s
        self._xm = 18 #nm
        self._DeltaG0 = 264.0 #pN.nm between F and U

        self.DeltaG0t = 300 #pN.nm  #barrier height without force between F and transition state
        self.xt = 9 #nm

        self.kB = 1.380649*1e-2 #pN.nm/K
        self._beta = 1/(self.kB*self._T) #pN^-1.nm^-1

        self._fc = self._DeltaG0/self._xm
        self.kc = self.kFtoU(self._fc)

        self._cached_results = {}  # Cache for memoization

    @property
    def DeltaG0(self):
        return self._DeltaG0
    @DeltaG0.setter
    def DeltaG0(self, val):
        self._DeltaG0 = val
        self._fc = val/self._xm
        self.kc = self.kFtoU(self._fc)
    
    @property
    def xm(self):
        return self._xm
    @xm.setter
    def xm(self, val):
        self._xm = val
        self._fc = self._DeltaG0/val
        self.kc = self.kFtoU(self._fc)
    
    @property
    def fc(self):
        return self._fc
    @fc.setter
    def fc(self, val):
        self._fc = val
        self._DeltaG0 = val*self._xm
        self.kc = self.kFtoU(val)
    
    @property
    def T(self):
        return self._T
    @T.setter
    def T(self, val):
        self._T = val
        self._beta = 1/(self.kB*val)

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, val):
        self._beta = val
        self._T = 1/(self.kB*val)
    
    
    def kFtoU(self, f:np.ndarray)->np.ndarray:
        return self.k0 * np.exp(- self.beta * (self.DeltaG0t - f*self.xt))

    def kUtoF(self, f:np.ndarray)->np.ndarray:
        return self.k0 * np.exp( - self.beta * (self.DeltaG0t - self.DeltaG0 + f*(self.xm - self.xt)))

    def DeltaG(self, f:np.ndarray)->np.ndarray:
        return self.DeltaG0 - f * self.xm
    
    def PsF(self, f1:np.ndarray, f2:np.ndarray, r:np.ndarray)->np.ndarray:
        """returns the survival probability of F from f1 to f2 at pull rate r in the forward process"""
        return np.exp(1 / (self.beta * self.xt * r) * (self.kFtoU(f1) - self.kFtoU(f2)))

    def PsU(self, f1:np.ndarray, f2:np.ndarray, r:np.ndarray)->np.ndarray:
        """returns the survival probability of U from f1 to f2 at pull rate r in the forward process"""
        return np.exp(1 / (self.beta * (self.xm - self.xt) * r) * (self.kUtoF(f2) - self.kUtoF(f1)))

    def avgWd(self, f_initial:np.ndarray, f_final:np.ndarray, r:np.ndarray)->np.ndarray:
        """returns the average dissipated work in the range [f_initial, f_final] when pulling at rate r
        in the forward process in the single-hopping approximation for U-type trajectories U->F->U"""

        force_list = np.linspace(8,22,N)
        Deltaf = force_list[1] - force_list[0]
        PsF_list = np.zeros((N,N), dtype=float)
        integral = np.zeros(N, dtype=float)
        for i in range(N):
            for j in range(N):
                PsF_list[i,j] = self.PsF(force_list[i], force_list[j], r)
                PsU_list[i,j] = self.PsU(force_list[i], force_list[j], r)
        PsF_int[i] = PsF_list[i,i:].sum()*Deltaf
        integral[i] = (self.kUtoF(force_list[i:])/r * PsU_list[i, i:]*PsF_int[i]).sum()*Deltaf

        return integral
    

    def kFtoU_mp(self, f:mp.mpf)->mp.mpf:
        return self.k0 * mp.exp( - self.beta * (self.DeltaG0t - f*self.xt))

    def kUtoF_mp(self, f:mp.mpf)->mp.mpf:
        return self.k0 * mp.exp( - self.beta * (self.DeltaG0t - self.DeltaG0 + f*(self.xm - self.xt)))
    
    def PsF_mp(self, f1:mp.mpf, f2:mp.mpf, r:mp.mpf)->mp.mpf:
        """returns the survival probability of F from f1 to f2 at pull rate r in the forward process"""
        return mp.exp(1 / (self.beta * self.xt * r) * (self.kFtoU_mp(f1) - self.kFtoU_mp(f2)))

    def PsU_mp(self, f1:mp.mpf, f2:mp.mpf, r:mp.mpf)->mp.mpf: #to change with xstar
        """returns the survival probability of U from f1 to f2 at pull rate r in the forward process"""
        return mp.exp(1 / (self.beta * (self.xm - self.xt) * r) * (self.kUtoF_mp(f2) - self.kUtoF_mp(f1)))

    def avgWd_int(self, f_initial:mp.mpf, f_final:mp.mpf, r:mp.mpf)->mp.mpf:
        """returns the average dissipated work in the range [f_initial, f_final] when pulling at rate r
        in the forward process in the single-hopping approximation for U-type trajectories U->F->U"""
        key = ('avgWd', f_initial, f_final, r)
        if key in self._cached_results:
            print('avgWd memo used')
        else:
            full_integral = mp.quad(lambda fprime: (self.kUtoF_mp(fprime) / r) * self.PsU_mp(f_initial, fprime, r) * mp.quad(lambda fdprime: self.PsF_mp(fprime, fdprime, r), [fprime, f_final]), [f_initial, f_final])
            self._cached_results[key] = (1 - self.PsU_mp(f_initial, f_final, r)) * full_integral * self.xm
        return self._cached_results[key]