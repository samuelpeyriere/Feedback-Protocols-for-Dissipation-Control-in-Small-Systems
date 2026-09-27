import numpy as np
from .builder import Protocol
from scipy.integrate import quad

class ArbitraryFeedback(Protocol):
    def __init__(self,
                 rF:float=4.0,
                 rU:float=17,
                 f1:float=13,
                 fMm1:float=18,
                 r:callable=None)->None:
        """
        Initialises the exponentially decreasing strategy.

        Parameters:
        - rF (float): the initial pull rate in pN/s
        - rU (float): the final pull rate in pN/s
        - f1 (float): the first measurement force in pN
        - fMm1 (float): the last measurement force in pN
        - r (callable): the function r(f)

        Returns:
        None
        """
        super().__init__()

        self.name = 'arbitrary_feedback'

        self.rU = rU
        self.fMm1 = fMm1

        self.observations = [float(f1)]
        self.rate_mask = [[float(rF)]]
    
        self.r = r

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

    def controlled_force_max_steps(self, f0:float, fM:float, Deltat:float)->int:
        self.f0 = f0
        self.fM = fM
        self.Deltat = Deltat
        self.half_interval_force = self.rF * (self.Deltat/2)
        return int((self.f1-self.f0)/(self.rF*self.Deltat) + quad(lambda f: 1/self.r(f), self.f1, self.fMm1)[0]/self.Deltat + (self.fM - self.fMm1)/(self.rU*self.Deltat)) + 10

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
        
        mask_to_r = (~state_list) & (force_list >= self.f1)
        mask_to_rU = (previous_rate_list == self.rU) | (state_list & (force_list >= self.f1)) | (force_list >= self.fMm1)
        
        updated_rate[mask_to_r] = self.r(force_list[mask_to_r])
        updated_rate[mask_to_rU] = self.rU
        
        to_return = np.empty_like(previous_rate_list, dtype=self.dtype)
        to_return['rate'] = updated_rate
        to_return['obs_idx'] = observation_index
        to_return['binary_hist'] = binary_history

        return to_return