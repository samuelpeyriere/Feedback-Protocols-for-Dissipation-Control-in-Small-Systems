import mpmath as mp
import numpy as np
from hairpyn.simulators.controlled_force import ControlledForceNumericalEstimation, ControlledForceSimulation
from hairpyn.protocol.none import NOProtocol
from hairpyn.protocol.DTF import DTFProtocol
from .builder import Protocol
from joblib import Parallel, delayed
from hairpyn.rng import seeded
from copy import copy

class CTFProtocol(Protocol):
    def __init__(self,
                 rF:float=5.0,
                 rU:float=17.0)->None:
        """
        Initialises the Continuous Time Feedback protocol

        Parameters:
        - rF (float): the initial pull rate in pN/s
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
        self.model = model
        if self.model.ensemble=='LdF':
            return int((self.fmax - self.fmin)/(self.rF*self.Deltat)) + 2
        elif self.model.ensemble=='FdL':
            return int((self.fmax - self.fmin + self.model.kx*self.model.xm)/(self.rF*self.Deltat)) + 2

    def updated_rate(self, observation_index:np.ndarray, binary_history:np.ndarray, state_list:np.ndarray, previous_rate_list:np.ndarray, force_list:np.ndarray, length_list:np.ndarray, forward:bool=True)->np.ndarray:
        """
        Takes the time, states, previous rates and forces of the system, returns the updated pull rates according to the CTF protocol.
    
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
        mask_to_rU = (previous_rate_list == self.rU) | (state_list == forward)

        if self.rU == np.inf:
            updated_rate[mask_to_rU] = (self.fmax - force_list[mask_to_rU] + 1e-10)/self.Deltat
        else:
            updated_rate[mask_to_rU] = self.rU
        
        to_return = np.empty_like(previous_rate_list, dtype=self.dtype)
        to_return['rate'] = updated_rate
        to_return['obs_idx'] = observation_index
        to_return['binary_hist'] = binary_history

        return to_return
    
    def Upsilon(self, outside_simulation, ratio_list:np.ndarray)->tuple:
        """
        Returns the thermodynamic information of the CTF protocol through brute force computation.

        Returns:
        np.ndarray: the Upsilon np.ndarray
        """
        
        simulation = copy(outside_simulation)
        mask_infty = np.isinf(ratio_list)

        f = np.linspace(simulation.fmin, simulation.fmax, 500)
        df= f[1] - f[0]
        bin_edges = np.concatenate([[f[0] - df/2], f[:-1] + df/2, [f[-1] + df/2]])

        simulation.forward = False
        simulation.initial_state = True
        simulation.N = 10000
        def computation(ratio): 
            simulation.protocol = NOProtocol(self.rF*ratio)
            simulation.run()
            pU_rU = np.mean(simulation.state_list, axis=1)
            bin_indices = np.digitize(simulation.force_list[:,0], bin_edges, right=True) #because NOProtocol
            simulation.reset()
            pU_rU_force = np.zeros_like(f)
            for i in range(1, len(bin_edges)):
                pU_rU_force[i-1] = np.mean(pU_rU[bin_indices == i])

            return np.log(df*np.sum(simulation.model.kUtoF(f)/self.rF*1/simulation.model.PsF(f, f[0], self.rF)*pU_rU_force)) #nansum gives wrong results
        
        Upsilon = np.zeros_like(ratio_list, dtype=float)
        Upsilon[~mask_infty] = np.array(Parallel(n_jobs=-1)(seeded(delayed(computation)(ratio) for ratio in ratio_list[~mask_infty])))
        Upsilon[mask_infty] = np.log(df*np.sum(simulation.model.kUtoF(f)/self.rF*1/simulation.model.PsF(f, f[0], self.rF)))

        simulation = None
        return Upsilon
    
    # def Upsilon(self, simulation, ratio_list:np.ndarray)->tuple:
    #     """
    #     Returns the thermodynamic information of the CTF protocol through brute force computation.

    #     Returns:
    #     np.ndarray: the Upsilon np.ndarray
    #     """
        
    #     forward_memory = simulation.forward
    #     initial_state_memory = simulation.initial_state
    #     N_memory = simulation.N
    #     protocol_memory = simulation.protocol

    #     simulation.forward = False
    #     simulation.initial_state = True
    #     simulation.N = 1000
        
    #     measurement_force_list = np.linspace(simulation.fmin, simulation.fmax, 500)
    #     diffs = np.diff(measurement_force_list)
    #     bin_edges = np.concatenate([
    #     [measurement_force_list[0] - diffs[0] / 2],  # First edge
    #     measurement_force_list[:-1] + diffs / 2,  # Middle edges
    #     [measurement_force_list[-1] + diffs[-1] / 2]  # Last edge
    #     ])
    
    #     def computation(ratio):
            
    #         rF = self.rF
    #         simulation.protocol = NOProtocol(rF)
    #         simulation.run()
    #         pU_rF = np.sum(simulation.state_list, axis=1)/simulation.N
            
    #         first_U_rF_flipped = np.argmax(simulation.state_list[::-1], axis=0)
    #         last_U_rF = simulation.state_list.shape[0] - 1 - first_U_rF_flipped
    #         phi_rF = np.zeros(simulation.state_list.shape[0])
    #         for elem in last_U_rF:
    #             phi_rF[elem] +=1
    #         phi_rF = phi_rF/simulation.N

    #         bin_indices = np.digitize(simulation.force_list[:,0], bin_edges, right=True) #because NOProtocol
    #         pU_rF_force = np.zeros_like(measurement_force_list)
    #         phi_rF_force = np.zeros_like(measurement_force_list)
    #         for i in range(1, len(bin_edges)):
    #             pU_rF_force[i-1] = np.mean(pU_rF[bin_indices == i])
    #             phi_rF_force[i-1] = np.mean(phi_rF[bin_indices == i])
            
    #         phi_rF_force = phi_rF_force*(simulation.fmax-simulation.fmin)/(simulation.protocol.rF*simulation.Deltat)/500

    #         rU = rF*ratio
    #         simulation.protocol = NOProtocol(rU)
    #         simulation.run()
    #         pU_rU = np.sum(simulation.state_list, axis=1)/simulation.N

    #         bin_indices = np.digitize(simulation.force_list[:,0], bin_edges, right=True)
    #         pU_rU_force = np.zeros(len(measurement_force_list))
    #         for i in range(1, len(bin_edges)):
    #             pU_rU_force[i-1] = np.mean(pU_rU[bin_indices == i])

    #         return np.log((pU_rU_force/(pU_rF_force+1e-15) * phi_rF_force).sum())

    #     Upsilon = np.array(Parallel(n_jobs=-1)(delayed(computation)(ratio) for ratio in ratio_list))
        
    #     simulation.forward = forward_memory
    #     simulation.initial_state = initial_state_memory
    #     simulation.N = N_memory
    #     simulation.protocol = protocol_memory
    
    #     return Upsilon
    
    # def Upsilon_single_hopping(self, numerical_estimation=ControlledForceNumericalEstimation())->mp.mpf:
    #     """returns the thermodynamic information of the Continuous Time Protocol"""
    #     return float(mp.log(1/self.rF * mp.quad(
    #         lambda f: mp.exp(-numerical_estimation.model.kUtoF_mp(f)/(numerical_estimation.model.beta*numerical_estimation.model.xt)
    #                          * (1/self.rU - 1/self.rF))
    #                          * 1/numerical_estimation.model.PsF_mp(f,numerical_estimation.fmin,self.rF) *numerical_estimation.model.kUtoF_mp(f)
    #                          * mp.exp(-numerical_estimation.model.kUtoF_mp(f)
    #                                     /(self.rF*numerical_estimation.model.beta*numerical_estimation.model.xt)),
    #                                       [numerical_estimation.fmin, numerical_estimation.fmax])))

    def DeltaWd_single_hopping(self, numerical_estimation=ControlledForceNumericalEstimation())->mp.mpf:
        """returns DeltaWdCTF the change in the average dissipated work upon implementing Continuous Time Feedback"""
        key = ('DeltaWdCTF', self.rF, self.rU)
        if key in numerical_estimation.model._cached_results:
            print('DeltaWdCTF memo used')
        else:
            numerical_estimation.model._cached_results[key] = mp.quad(lambda fprime: numerical_estimation.model.PsF_mp(numerical_estimation.fmin, fprime, self.rF) * numerical_estimation.model.kFtoU_mp(fprime)/self.rF * (numerical_estimation.model.avgWd(fprime, numerical_estimation.fmax, self.rF) - numerical_estimation.model.avgWd(fprime, numerical_estimation.fmax, self.rU)), [numerical_estimation.fmin, numerical_estimation.fmax])
        return float(numerical_estimation.model._cached_results[key])
    
    def avg_time(self, simulation, ratio_list:np.ndarray)->float:
        """returns the estimated average measurement time"""
        f = np.linspace(simulation.fmin, simulation.fmax, 1000)
        Deltaf = f[1] - f[0]
        avgT = np.zeros_like(ratio_list)
        for k, ratio in np.ndenumerate(ratio_list):
            avgT[k] = np.sum(simulation.model.kFtoU(f)/self.rF * simulation.model.PsF(f[0], f, self.rF) * ((f-f[0])/self.rF + (simulation.fmax - f)/(ratio*self.rF)))*Deltaf
        return avgT

    
    def DeltaWd(self, outside_simulation, ratio_list:np.ndarray)->float:
        """returns the estimated dissipated work"""
        
        simulation = copy(outside_simulation)
        simulation.N = 3000

        mask_infty = np.isinf(ratio_list)

        f = np.linspace(simulation.fmin, simulation.fmax, 50)
        df= f[1] - f[0]

        Wd_list_rF = np.zeros_like(f)
        simulation.protocol = NOProtocol(self.rF)
        for k, f_ in np.ndenumerate(f):
            simulation.fmin = f_
            simulation.initial_state = True
            simulation.run(pre_thermalise=False)
            Wd_list_rF[k] = simulation.dissipated_work().mean()
            simulation.reset()
        
        def computation(ratio):
            
            Wd_list_rU = np.zeros_like(f)
            simulation.protocol = NOProtocol(self.rF*ratio)
            for k, f_ in np.ndenumerate(f):
                simulation.fmin = f_
                simulation.initial_state = True
                simulation.run(pre_thermalise=False)
                Wd_list_rU[k] = simulation.dissipated_work().mean()
                simulation.reset()

            return np.sum(simulation.model.kFtoU(f)/self.rF * simulation.model.PsF(f[0], f, self.rF)*(Wd_list_rF - Wd_list_rU))*df

        DeltaWd = np.zeros_like(ratio_list, dtype=float)
        DeltaWd[~mask_infty] = np.array(Parallel(n_jobs=-1)(seeded(delayed(computation)(ratio) for ratio in ratio_list[~mask_infty])))
        DeltaWd[mask_infty] = np.sum(simulation.model.kFtoU(f)/self.rF * simulation.model.PsF(f[0], f, self.rF)*Wd_list_rF)*df

        simulation = None
        return DeltaWd

    def Upsilon_sh(self, outside_simulation, ratio_list):
        f_list = np.linspace(outside_simulation.initial_force, outside_simulation.final_force,1000)
        Deltaf = f_list[1] - f_list[0]
        PsF_list_rF = outside_simulation.model.PsF(f_list,outside_simulation.initial_force,self.rF)
        return np.array(Parallel(n_jobs=-1)(delayed(lambda ratio: np.log(np.nansum(1/PsF_list_rF * outside_simulation.model.kUtoF(f_list) * 1/outside_simulation.model.PsU(outside_simulation.initial_force,f_list,ratio*self.rF)) * Deltaf/self.rF))(ratio) for ratio in ratio_list))

    def avg_time_sh(self, outside_simulation, ratio_list):
        f_list = np.linspace(outside_simulation.initial_force, outside_simulation.final_force,1000)
        Deltaf = f_list[1] - f_list[0]
        return np.array(Parallel(n_jobs=-1)(delayed(lambda ratio: (outside_simulation.model.kFtoU(f_list)/self.rF*outside_simulation.model.PsF(outside_simulation.initial_force,f_list,self.rF)*((f_list-outside_simulation.initial_force)/self.rF + (outside_simulation.final_force-f_list)/(ratio*self.rF))).sum()*Deltaf)(ratio) for ratio in ratio_list))