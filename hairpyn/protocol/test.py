import mpmath as mp
import numpy as np
from hairpyn.simulators.controlled_force import ControlledForceNumericalEstimation, ControlledForceSimulation
from .builder import Protocol
from joblib import Parallel, delayed
from copy import copy

class TestProtocol(Protocol):
    def __init__(self,
                 rF:float=5.0,
                 rU:float=17.0)->None:
        """
        Initialises the Discrete Time Feedback protocol

        Parameters:
        - rF (float): the initial pull rate in pN/s
        - f1 (float): the measurement force in pN
        - rU (float): the final pull rate in pN/s

        Returns:
        None
        """
        super().__init__()
        self.name = 'CTF'
        self.observations = [[float(8),float(22)]]
        self.rate_mask = [[float(rF)], [np.nan, float(rU)]]

    @property
    def rF(self):
        return self.rate_mask[0][0]
    @rF.setter
    def rF(self, val):
        self.rate_mask[0][0] = float(val)

    @property
    def rU(self):
        return self.rate_mask[1][1]
    @rU.setter
    def rU(self, val):
        self.rate_mask[1][1] = float(val)
        
    def max_steps(self, model, fmin:float, fmax:float, Deltat:float)->int:
        self.fmax = fmax
        self.fmin = fmin
        self.Deltat = Deltat
        if model.ensemble=='LdF':
            return int((self.fmax - self.fmin)/(self.rF*self.Deltat)) + 2
        elif model.ensemble=='FdL':
            return int((self.fmax - self.fmin + model.kx*model.xm)/(self.rF*self.Deltat)) + 2
    
    def updated_rate(self, model, observation_index:np.ndarray, binary_history:np.ndarray, state_list:np.ndarray, previous_rate_list:np.ndarray, force_list:np.ndarray, forward:bool=True)->np.ndarray:
        """
        Takes the model, time, states, previous rates and forces of the system, returns the updated pull rates according to the CTF protocol.
    
        Parameters:
        - model (class): the class containing the model parameters
        - time (float): the time at that step
        - state_list (ndarray): The hairpin's states (Folded=False or Unfolded=True)
        - previous_rate_list (ndarray): The previous pulling rates (pN/s)
        - force_list (ndarray): The forces pulling on the systems (pN)
        - forward (bool): The pull direction on the system

        Returns:
        ndarray: The updated pull rates
        """            
        updated_rate = np.full_like(previous_rate_list, self.rF)
        mask_to_rU = (previous_rate_list == self.rU) | (state_list == forward)
        updated_rate[mask_to_rU] = (self.fmax - force_list[mask_to_rU] + 1e-10)/self.Deltat if self.rU == np.inf else self.rU 
  
        to_return = np.empty_like(previous_rate_list, dtype=self.dtype)
        to_return['rate'] = updated_rate
        to_return['obs_idx'] = observation_index
        to_return['binary_hist'] = binary_history

        return to_return
