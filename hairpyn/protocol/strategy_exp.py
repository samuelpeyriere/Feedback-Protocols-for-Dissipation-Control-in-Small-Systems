import numpy as np
from .builder import Protocol
from scipy.optimize import root_scalar

class StrategyExp(Protocol):
    def __init__(self,
                 rF:float=4.0,
                 f1:float=13,
                 fini:float=12,
                 fmaxi:float=17,
                 fscale:float=3,
                 rU:float=17)->None:
        """
        Initialises the exponentially decreasing strategy.

        Parameters:
        - rF (float): the initial pull rate in pN/s
        - f1 (float): the measurement force in pN
        - fscale (float): the exponential scaling parameter in pN
        - rU (float): the final pull rate in pN/s

        Returns:
        None
        """
        super().__init__()

        self.name = 'strategy_exp'
        self.fscale = fscale
        self.rU = rU
        self._fini = fini
        self.fmaxi = fmaxi

        self.observations = [float(f1)]
        self.rate_mask = [[float(rF)]]
    
        self._r = lambda f: self.rF*np.exp(-(f-self.fini)/self.fscale)

    @property
    def r(self):
        return self._r
    @r.setter
    def r(self, func):
        self._r = func
        self._fini = root_scalar(lambda f: func(f)-self.rF, bracket=[8,22]).root
    
    @property
    def f1(self):
        return self.observations[0]
    @f1.setter
    def f1(self, val):
        self.observations[0] = float(val)
    
    @property
    def rF(self):
        return self.rate_mask[0][0]
    @rF.setter
    def rF(self, val):
        self.rate_mask[0][0] = float(val)
        self._fini = root_scalar(lambda f: self.r(f)-val, bracket=[8,22]).root

    # @property
    # def rU(self):
    #     return self.rate_mask[1][1]
    # @rU.setter
    # def rU(self, val):
    #     self.rate_mask[1][1] = float(val)
    
    @property
    def fini(self):
        return self._fini
    @fini.setter
    def fini(self, val):
        self._fini = val
    

    def controlled_force_max_steps(self, fmin:float, fmax:float, Deltat:float)->int:
        self.fmax = fmax
        self.fmin = fmin
        self.Deltat = Deltat
        self.half_interval_force = self.rF * (self.Deltat/2)
        self.half_interval_length = 0.6 #nm
        f_list_integral = np.linspace(self.fini, self.fmaxi, 1000)
        Deltaf = f_list_integral[1] - f_list_integral[0]
        return int((self.f1-self.fmin)/(self.rF*self.Deltat) + (self.fmax - self.fmaxi)/(self.rU*self.Deltat) + (Deltaf/(self.r(f_list_integral)*self.Deltat)).sum()) + 10

    def controlled_force_updated_rate(self, observation_index, binary_history, state_list:np.ndarray, previous_rate_list:np.ndarray, force_list:np.ndarray, forward:bool=True)->np.ndarray:
        """
        Takes the time, states, previous rates and forces of the system, returns the updated pull rates according to the DTF+CTF strategy.
    
        Parameters:
        - time (float): the time at that step
        - state_list (ndarray): The hairpin's states (Folded=False or Unfolded=True)
        - previous_rate_list (ndarray): The previous pulling rates (pN/s)
        - force_list (ndarray): The forces pulling on the systems (pN)
        - forward (bool): The pull direction on the system

        Returns:
        ndarray: The updated pull rates
        """
        updated_rate = np.full_like(previous_rate_list, self.rF)
        
        mask_to_rexp = (~state_list) & (force_list > self.f1) & (force_list > self.fini)
        mask_to_rU = (previous_rate_list == self.rU) | ((state_list==forward) & (force_list >= self.f1)) | (force_list > self.fmaxi)
        
        updated_rate[mask_to_rexp] = self.r(force_list[mask_to_rexp])
        updated_rate[mask_to_rU] = self.rU
        

        to_return = np.empty_like(previous_rate_list, dtype=self.dtype)
        to_return['rate'] = updated_rate
        to_return['obs_idx'] = observation_index
        to_return['binary_hist'] = binary_history

        return to_return