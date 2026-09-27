import numpy as np

from hairpyn.simulators.simulator import Simulation
from hairpyn.protocol.none import NOProtocol
from .builder import Protocol

from joblib import Parallel, delayed

class DTFProtocol(Protocol):
    def __init__(self,
                 rF:float=5.0,
                 rU:float=17.0,
                 cp1:float=14.0)->None:
        """
        Initialises the Discrete Time Feedback Protocol

        Parameters:
        - rF (float): the initial pull rate in pN/s
        - rU (float): the final pull rate in pN/s
        - cp1 (float): the control parameter (measurement force in pN or length in nm)

        Returns:
        None
        """
        super().__init__()
        self.name = 'DTF'
        self.observations = [float(cp1)]
        self.rate_mask = [[float(rF)], [np.nan, float(rU)]]

    @property
    def f1(self):
        return self.cp1
    @f1.setter
    def f1(self, val):
        self.cp1 = float(val)

    @property
    def l1(self):
        return self.cp1
    @l1.setter
    def l1(self, val):
        self.cp1 = float(val)
    
    @property
    def cp1(self):
        return self.observations[0]
    @cp1.setter
    def cp1(self, val):
        self.observations[0] = float(val)
    
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
        self.model = model
        self.fmin = fmin
        self.fmax = fmax
        self.Deltat = Deltat
        if self.model.ensemble=='LdF':
            self.half_interval = self.rF * (self.Deltat/2)
            return int((self.fmax - self.fmin)/(self.rF*self.Deltat)) + 2
        elif self.model.ensemble=='FdL':
            self.half_interval = self.rF * (self.Deltat/2)/model.kx
            return int((self.fmax - self.fmin + model.kx*model.xm)/(self.rF*self.Deltat)) + 2
        
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

        updated_rate = np.full_like(previous_rate_list, self.rF)
        mask_to_rU = (previous_rate_list == self.rU) | ((state_list==forward) & (self.cp1-self.half_interval < cp_list) & (cp_list <= self.cp1+self.half_interval))
        if self.rU == np.inf:
            updated_rate[mask_to_rU] = (self.fmax - force_list[mask_to_rU] + 1e-10)/self.Deltat
        else:
            updated_rate[mask_to_rU] = self.rU

        to_return = np.empty_like(previous_rate_list, dtype=self.dtype)
        to_return['rate'] = updated_rate
        to_return['obs_idx'] = observation_index
        to_return['binary_hist'] = binary_history

        return to_return
    
    def DeltaWd(self, outside_simulation, cp1_list:np.ndarray)->np.ndarray:
        """
        Returns the average dissipated work of the DTF protocol by decomposing into simpler simulations.

        Parameters:
        - outside_simulation (ndarray): simulation object that holds the simulation parameters to use
        - f1_list (ndarray): the considered values for f1
        
        Returns:
        np.ndarray: The average dissipated work
        """

        simulation = Simulation(N=10000, forward=True)

        simulation.model = outside_simulation.model
        simulation.fmax = outside_simulation.fmax
        simulation.fmin = outside_simulation.fmin
        simulation.Deltat = outside_simulation.Deltat
        
        if simulation.model.ensemble=='LdF':
            def measurement_index(measurement_cp, rate):
                return int(abs(simulation.initial_force - measurement_cp)/(rate*simulation.Deltat))
        elif simulation.model.ensemble=='FdL':
            def measurement_index(measurement_cp, rate):
                return int(abs(simulation.initial_length - measurement_cp)/(rate*simulation.Deltat/simulation.model.kx))
        
        simulation.protocol = NOProtocol(rF=self.rF)
        simulation.run()
        pU = np.sum(simulation.state_list, axis=1)/simulation.N
        pU_cp = np.zeros_like(cp1_list)
        for k, measurement_cp in np.ndenumerate(cp1_list):
            pU_cp[k] = pU[measurement_index(measurement_cp, simulation.protocol.rF)]

        simulation.fmax = outside_simulation.fmax
        simulation.initial_state = True
        
        if simulation.model.ensemble=='LdF':
            def fmin(cp1):
                return cp1
        elif simulation.model.ensemble=='FdL':
            def fmin(cp1):
                return simulation.model.force(True, cp1)

        def computation(cp1):
            simulation.fmin = fmin(cp1)
            simulation.protocol = NOProtocol(rF=self.rF)
            simulation.run(pre_thermalise=False)
            _Wd_rF = simulation.dissipated_work().mean()
            simulation.reset()

            if self.rU == np.inf:
                _Wd_rU = 0
            else:
                simulation.protocol = NOProtocol(rF=self.rU)
                simulation.run(pre_thermalise=False)
                _Wd_rU = simulation.dissipated_work().mean()
                simulation.reset()

            return _Wd_rF, _Wd_rU
        
        Wd_rF, Wd_rU = map(np.array, zip(*Parallel(n_jobs=-1)(delayed(computation)(cp1) for cp1 in cp1_list)))

        return pU_cp * (Wd_rF - Wd_rU - (Wd_rF - Wd_rU)[-1])

    def avg_time(self, outside_simulation, cp1_list:np.ndarray)->np.ndarray:
        """
        Returns the average time of the protocol by decomposing into simpler simulations.

        Parameters:
        - outside_simulation (ndarray): simulation object that holds the simulation parameters to use
        - f1_list (ndarray): the considered values for f1
        
        Returns:
        np.ndarray: The average time of the protocol
        """

        simulation = Simulation(N=10000, forward=True)
        simulation.model = outside_simulation.model
        simulation.fmin = outside_simulation.fmin
        simulation.Deltat = outside_simulation.Deltat

        if simulation.model.ensemble=='LdF':
            def measurement_index(measurement_cp, rate):
                return int(abs(simulation.initial_force - measurement_cp)/(rate*simulation.Deltat))
        elif simulation.model.ensemble=='FdL':
            def measurement_index(measurement_cp, rate):
                return int(abs(simulation.initial_length - measurement_cp)/(rate/simulation.model.kx*simulation.Deltat))
        
        simulation.protocol = NOProtocol(rF=self.rF)
        simulation.run()
        pU = np.sum(simulation.state_list, axis=1)/simulation.N
        pU_cp = np.zeros_like(cp1_list)
        for k, measurement_cp in np.ndenumerate(cp1_list):
            pU_cp[k] = pU[measurement_index(measurement_cp, self.rF)]

        if simulation.model.ensemble=='LdF':
            return (cp1_list - outside_simulation.fmin)/self.rF + (outside_simulation.fmax - cp1_list)*(pU_cp/self.rU + (1-pU_cp)/self.rF)
        elif simulation.model.ensemble=='FdL':
            return (cp1_list - simulation.model.length(False, outside_simulation.fmin))/self.rF + (simulation.model.length(True, outside_simulation.fmax) - cp1_list)*(pU_cp/self.rU + (1-pU_cp)/self.rF)*simulation.model.kx

    def Upsilon(self, outside_simulation, cp1_list:np.ndarray)->np.ndarray:
        """
        Returns the thermodynamic information Upsilon of the protocol by decomposing into simpler simulations.

        Parameters:
        - outside_simulation (ndarray): simulation object that holds the simulation parameters to use
        - cp1_list (ndarray): the considered values for cp1
        
        Returns:
        np.ndarray: The computed Upsilon
        """

        simulation = Simulation(N=10000, forward=False)

        simulation.model = outside_simulation.model
        simulation.fmin = outside_simulation.fmin
        simulation.fmax = outside_simulation.fmax
        simulation.Deltat = outside_simulation.Deltat
        
        if simulation.model.ensemble=='LdF':
            def measurement_index(measurement_cp, rate):
                return int(abs(simulation.initial_force - measurement_cp)/(rate*simulation.Deltat))
        elif simulation.model.ensemble=='FdL':
            def measurement_index(measurement_cp, rate):
                return int(abs(simulation.initial_length - measurement_cp)/(rate*simulation.Deltat/simulation.model.kx))
        
        simulation.protocol = NOProtocol(rF=self.rF)
        simulation.run()
        pF = np.sum(~simulation.state_list, axis=1)/simulation.N
        pF_cp = np.zeros_like(cp1_list)
        for k, measurement_cp in np.ndenumerate(cp1_list):
            pF_cp[k] = pF[measurement_index(measurement_cp, self.rF)]
        
        if self.rU == np.inf:
            pU_cp = np.ones_like(cp1_list)
        else:
            simulation.protocol = NOProtocol(rF=self.rU)
            simulation.run()
            pU = np.sum(simulation.state_list, axis=1)/simulation.N
            pU_cp = np.zeros_like(cp1_list)
            for k, measurement_cp in np.ndenumerate(cp1_list):
                pU_cp[k] = pU[measurement_index(measurement_cp, self.rU)]

        simulation.reset()
    
        return np.log(pU_cp + pF_cp)

    def Upsilon_sh(self, outside_simulation, cp_list:np.ndarray)->np.ndarray:
        if outside_simulation.model.ensemble == 'LdF':
            f_list = cp_list
        elif outside_simulation.model.ensemble == 'FdL':
            f_list = outside_simulation.model.force(True, cp_list)
        return np.log(1/outside_simulation.model.PsU(outside_simulation.final_force,f_list,self.rU) + 1 - 1/outside_simulation.model.PsU(outside_simulation.final_force,f_list,self.rF))
    
    def avg_time_sh(self, outside_simulation, cp_list:np.ndarray)->np.ndarray:
        if outside_simulation.model.ensemble == 'LdF':
            return (cp_list - outside_simulation.initial_force)/self.rF + (outside_simulation.final_force - cp_list)*(1/self.rU + (1/self.rF - 1/self.rU)*outside_simulation.model.PsF(outside_simulation.initial_force, cp_list, self.rF))
        elif outside_simulation.model.ensemble == 'FdL':
            return (cp_list - outside_simulation.initial_length)*outside_simulation.model.kx/self.rF + (outside_simulation.final_length - cp_list)*outside_simulation.model.kx*(1/self.rU + (1/self.rF - 1/self.rU)*outside_simulation.model.PsF(outside_simulation.initial_force, outside_simulation.model.force(False, cp_list), self.rF))