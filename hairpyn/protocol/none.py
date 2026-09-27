import mpmath as mp
import numpy as np
from .builder import Protocol

class NOProtocol(Protocol):
    def __init__(self,
                 rF:float=5.0,
                 thermalisation_time:float=1)->None:
        """
        Initialises the constant pull rate protocol.

        Parameters:
        - rF (float): the constant pull rate in pN/s

        Returns:
        None
        """
        super().__init__()
        self.name = 'none'
        self.observations = []
        self.rate_mask = [[float(rF)]]
        self.thermalisation_time = thermalisation_time

    @property
    def rF(self):
        return self.rate_mask[0][0]
    @rF.setter
    def rF(self, val):
        self.rate_mask = [[float(val)]]
    
    @property
    def initial_rate(self):
        return self.rate_mask[0][0]
    @initial_rate.setter
    def initial_rate(self, val):
        self.rate_mask = [[float(val)]]

    def max_steps(self, model, fmin:float, fmax:float, Deltat:float)->int:
        self.fmax = fmax
        self.fmin = fmin
        self.Deltat = Deltat
        self.model = model
        if self.model.ensemble=='LdF':
            return int(self.thermalisation_time/self.Deltat) if self.rF==0 else int((self.fmax - self.fmin)/(self.rF*self.Deltat)) + 2
        elif self.model.ensemble=='FdL':
            return int(self.thermalisation_time/self.Deltat) if self.rF==0 else int((self.fmax - self.fmin + model.kx*model.xm)/(self.rF*self.Deltat)) + 2

    def updated_rate(self, observation_index:np.ndarray, binary_history:np.ndarray, state_list:np.ndarray, previous_rate_list:np.ndarray, force_list:np.ndarray, length_list:np.ndarray, forward:bool=True)->np.ndarray:
        """
        Takes the time, states, previous rates and forces of the system, returns the updated pull rates without protocol.
    
        Parameters:
        - time (float): the time at that step
        - state_list (ndarray): The hairpin's states (Folded=False or Unfolded=True)
        - previous_rate_list (ndarray): The previous pulling rates (pN/s)
        - force_list (ndarray): The forces pulling on the systems (pN)
        - forward (bool): The pull direction on the system

        Returns:
        ndarray: The updated pull rates
        """     
        to_return = np.empty_like(previous_rate_list, dtype=self.dtype)
        to_return['rate'] =  np.full_like(previous_rate_list, self.rF, dtype=float)
        to_return['obs_idx'] = observation_index
        to_return['binary_hist'] = binary_history

        return to_return