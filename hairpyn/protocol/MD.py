import numpy as np
from .builder import Protocol

class MDProtocol(Protocol):
    def __init__(self, f0:float=14, Deltaf:float=1, rQS:float=0.01):

        super().__init__()
        
        self.name = 'MD'
        self.rP = np.inf
        self.rQS = rQS #Quasi static pull rate pN/s

        self.observations = [float(f0), float(f0+Deltaf), float(f0-Deltaf)]
        self.rate_mask = [[0], [self.rP, -self.rP], [-self.rQS for _ in range(4)], [self.rQS for _ in range(8)]]
    
    @property
    def f0(self):
        return self.observations[0]
    @f0.setter
    def f0(self, val):
        Deltaf = self.Deltaf
        self.observations[0] = float(val)
        self.observations[1] = float(val) + Deltaf
        self.observations[2] = float(val) - Deltaf
    
    @property
    def Deltaf(self):
        return self.observations[1] - self.observations[0]
    @Deltaf.setter
    def Deltaf(self, val):
        self.observations[1] = self.f0 + float(val)
        self.observations[2] = self.f0 - float(val)

    def max_steps(self, model, fmin:float, fmax:float, Deltat:float)->int:
        self.model = model
        self.Deltat = Deltat
        if self.rP == np.inf:
            self.rP = self.Deltaf/self.Deltat
        self.half_interval = self.rQS * (self.Deltat/2)
        return int(self.Deltaf * (1/self.rP + 1/self.rQS) / self.Deltat) + 3

    def updated_rate(self, observation_index:np.ndarray, binary_history:np.ndarray, state_list:np.ndarray, previous_rate_list:np.ndarray, force_list:np.ndarray, length_list:np.ndarray, forward:bool=True)->np.ndarray:
        """
        Takes the observation index, binary history, states, previous rates and forces of the system, returns the updated pull rates according to the DTF protocol.
    
        Parameters:
        - observation_index (ndarray): the index in observation of the current measurement
        - binary_history (ndarray): the recorded history of previous observations as a base 10 binary number
        - state_list (ndarray): The hairpin's states (Folded=False or Unfolded=True)
        - previous_rate_list (ndarray): The previous pulling rates (pN/s)
        - force_list (ndarray): The forces pulling on the systems (pN)
        - forward (bool): The pull direction on the system
        
        Returns:
        ndarray: The updated pull rates
        """
        
        if self.model.ensemble=='LdF':
            cp_list = force_list
        elif self.model.ensemble=='FdL':
            cp_list = length_list

        updated_rate = np.zeros_like(previous_rate_list)

        maskf0 = (cp_list >= self.f0 - self.half_interval) & (cp_list <= self.f0 + self.half_interval)
        maskDeltaf = (np.abs(self.f0 - cp_list) >= self.Deltaf-self.half_interval) & (np.abs(self.f0 - cp_list) <= self.Deltaf+self.half_interval)
        maskrP = (np.abs(previous_rate_list) == self.rP)

        maskrQS = (np.abs(previous_rate_list) == self.rQS)
        masksign= (previous_rate_list > 0)
        
        updated_rate[(~maskf0)*maskrP*masksign] = + self.rP
        updated_rate[(~maskf0)*maskrP*(~masksign)] = - self.rP

        updated_rate[(~maskf0)*maskrQS*masksign] = + self.rQS
        updated_rate[(~maskf0)*maskrQS*(~masksign)] = - self.rQS

        updated_rate[maskf0*state_list*(previous_rate_list==0)] = + self.rP
        updated_rate[maskf0*(~state_list)*(previous_rate_list==0)] = - self.rP

        updated_rate[maskDeltaf*masksign*maskrP] = - self.rQS
        updated_rate[maskDeltaf*(~masksign)*maskrP] = + self.rQS

        updated_rate[maskf0*maskrQS] = np.nan #ends cycle
        
        to_return = np.empty_like(previous_rate_list, dtype=self.dtype)
        to_return['rate'] = updated_rate
        to_return['obs_idx'] = observation_index
        to_return['binary_hist'] = binary_history

        return to_return
    
    
