import mpmath as mp
import numpy as np
from hairpyn.protocol import NOProtocol
from hairpyn.simulators.simulator import Simulation
from .builder import Protocol
from joblib import Parallel, delayed
from copy import copy

class Strategy(Protocol):
    def __init__(self,
                 rF:float=4.0,
                 rFprime:float=1.0,
                 rU:float=17.0,
                 cp1:float=14.0)->None:
        """
        Initialises the strategy.

        Parameters:
        - rF (float): the initial pull rate in pN/s
        - rFprime (float): the slower pull rate in pN/s
        - rU (float): the final pull rate in pN/s
        - cp1 (float): the control parameter

        Returns:
        None
        """
        super().__init__()
        self.name = 'strategy'
        self.observations = [float(cp1), [float(cp1),float(22)]]
        self.rate_mask = [[float(rF)], [float(rFprime), float(rU)], [np.nan, float(rU), np.nan, np.nan]]
    
    @property
    def cp1(self):
        return self.observations[0]
    @cp1.setter
    def cp1(self, val):
        self.observations[0] = float(val)
        self.observations[1][0] = float(val)
    
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
    def rF(self):
        return self.rate_mask[0][0]
    @rF.setter
    def rF(self, val):
        self.rate_mask[0][0] = float(val)

    @property
    def rFprime(self):
        return self.rate_mask[1][0]
    @rFprime.setter
    def rFprime(self, val):
        self.rate_mask[1][0] = float(val)

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
            if self.rF == np.inf:
                self.half_interval = 1e-10
            else:
                self.half_interval = self.rF * (self.Deltat/2)
            return int((self.f1-self.fmin)/(self.rF*self.Deltat) + (self.fmax - self.f1)/(self.rFprime*self.Deltat)) + 100
        elif self.model.ensemble=='FdL':
            self.half_interval = self.rF * (self.Deltat/2)/self.model.kx     
            return int((self.model.force(False, self.cp1)-self.fmin)/(self.rF*self.Deltat) + (self.fmax - self.model.force(False, self.cp1) + self.model.kx*self.model.xm)/(self.rFprime*self.Deltat)) + 100    
    
    def updated_rate(self, observation_index:np.ndarray, binary_history:np.ndarray, state_list:np.ndarray, previous_rate_list:np.ndarray, force_list:np.ndarray, length_list:np.ndarray, forward:bool=True)->np.ndarray:
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
             
        if self.model.ensemble=='LdF':
            cp_list = force_list
        elif self.model.ensemble=='FdL':
            cp_list = length_list

        updated_rate = np.full_like(previous_rate_list, self.rF)
        
        mask_around_cp1 = (self.cp1 - self.half_interval < cp_list) & (cp_list <= self.cp1 + self.half_interval)
        mask_to_rFprime = ((~state_list == forward) & mask_around_cp1) | (previous_rate_list == self.rFprime)
        mask_to_rU = (previous_rate_list == self.rU) | ((state_list == forward) & (cp_list > self.cp1 + self.half_interval)) 
        
        updated_rate[mask_to_rFprime] = self.rFprime
 
        if self.rU == np.inf:
            updated_rate[mask_to_rU] = (self.fmax - force_list[mask_to_rU] + 1e-10)/self.Deltat
        else:
            updated_rate[mask_to_rU] = self.rU
        
        to_return = np.empty_like(previous_rate_list, dtype=self.dtype)
        to_return['rate'] = updated_rate
        to_return['obs_idx'] = observation_index
        to_return['binary_hist'] = binary_history

        return to_return
    
    
    def Upsilon(self, outside_simulation, f1_list)->tuple:
        """
        Returns the thermodynamic information of the considered protocol through brute force computation.

        Returns:
        tuple: the force np.ndarray/ratio np.ndarray and the Upsilon np.ndarray
        """

        simulation = copy(outside_simulation)
        
        f = np.linspace(simulation.fmin, simulation.fmax, 500) #could directly run simulation.
        df = f[1] - f[0]
        bin_edges = np.concatenate([[f[0]-df/2], f[:-1]+df/2, [f[-1]+df/2]])
        
        X,Y = np.meshgrid(f, f)
        PsF_list_rFprime = simulation.model.PsF(Y,X,self.rFprime)

        if self.rU == np.inf:
            pU_f = np.ones_like(f, dtype=float)
        else:
            simulation.N = 1000
            simulation.protocol = NOProtocol(self.rU)
            simulation.forward = False
            simulation.initial_state = True
            #simulation.fmin = np.min(f1_list)
            simulation.run()
            pU = np.sum(simulation.state_list, axis=1)/simulation.N
            bin_indices = np.digitize(simulation.force_list[:,0], bin_edges, right=True) #Return the indices of the bins to which each value in input array belongs.
            simulation.reset()
            pU_f = np.zeros_like(f, dtype=float)
            for i in range(1, len(bin_edges)):
                if len(pU[bin_indices == i])==0:
                    pU_f[i-1] = 0
                else:
                    pU_f[i-1] = np.mean(pU[bin_indices == i])

        Upsilon = np.zeros_like(f1_list, dtype=float)
        for k, f1 in np.ndenumerate(f1_list):
            idx = np.argmin(np.abs(f - f1))
            Upsilon[k] = np.log(pU_f[idx] + (simulation.model.kUtoF(f[idx:])*1/PsF_list_rFprime[idx:,idx]*pU_f[idx:]).sum()*df/self.rFprime)
        
        simulation = None
        return Upsilon

    def avg_time(self, outside_simulation, f1_list:np.ndarray)->float:
        """returns the estimated average measurement time"""
        simulation = copy(outside_simulation)
        
        f = np.linspace(np.min(f1_list), simulation.fmax, 1000)
        df = f[1] - f[0]
        bin_edges = np.concatenate([[f[0]-df/2], f[:-1]+df/2, [f[-1]+df/2]])
        
        simulation.N = 5000
        simulation.protocol = NOProtocol(self.rF)
        simulation.fmax = np.max(f1_list)
        simulation.run()
        pU = np.mean(simulation.state_list, axis=1) #-1 because of +1 in the simulation.run() ?
        bin_indices = np.digitize(simulation.force_list[:,0], bin_edges, right=True)
        simulation.reset()
        simulation.fmax = outside_simulation.fmax
        pU_f = np.zeros_like(f, dtype=float)
        for i in range(1, len(bin_edges)):
            pU_f[i-1] = np.mean(pU[bin_indices == i])
        
        X,Y = np.meshgrid(f, f)
        PsF_list_rFprime = simulation.model.PsF(Y,X,self.rFprime)
        
        avgT = np.zeros_like(f1_list, dtype=float)
        for k, f1 in np.ndenumerate(f1_list):
            idx = np.argmin(np.abs(f - f1)) #doesn't count the additional point introduced because of +1 in simulation.run()
            avgT[k] = (f1 - simulation.fmin)/self.rF + pU_f[idx]*(simulation.fmax - f1)/self.rU + (1-pU_f[idx])*(simulation.model.kFtoU(f[idx:])*PsF_list_rFprime[idx,idx:]*((f[idx:] - f[idx])/self.rFprime + (simulation.fmax - f[idx:])/self.rU)).sum()*df/self.rFprime
        
        simulation = None
        return avgT

    # def Upsilon_single_hopping(self, numerical_estimation=ControlledForceNumericalEstimation())->float:
    #     """returns the thermodynamic information of the protocol strategy""" #approximation works for M>20 and rF>4
    #     f = mp.linspace(self.f1, numerical_estimation.fmax, numerical_estimation.M)
    #     df = f[1]-f[0]
    #     return float(mp.log(mp.exp(-numerical_estimation.model.kUtoF_mp(self.f1)/(numerical_estimation.model.beta*numerical_estimation.model.xt*self.rU)) + df/self.rFprime * sum((mp.exp(-numerical_estimation.model.kUtoF_mp(f[k])/(numerical_estimation.model.beta*numerical_estimation.model.xt)*(1/self.rU - 1/self.rFprime))* numerical_estimation.model.kUtoF_mp(f[k]) * 1/numerical_estimation.model.PsU_mp(numerical_estimation.fmax, f[k], self.rFprime) for k in range(1,numerical_estimation.M)))))
    
    # def DeltaWd_single_hopping(self, numerical_estimation=ControlledForceNumericalEstimation())->float:
    #     """returns DeltaWdstrategy the change in the average dissipated work upon implementing the Feedback strategy
    #     with M-1 intermediate measurements including f1, neglecting the terms in WdrF and WdrFprime"""
    #     key = ('DeltaWdstrategy', self.f1, numerical_estimation.M, self.rF, self.rFprime, self.rU)
    #     if key in numerical_estimation.model._cached_results:
    #         print('DeltaWdstrategy memo used')
    #         return numerical_estimation.model._cached_results[key]

    #     protocol = DTFProtocol(rF=self.rFprime, rU=self.rU, f1=self.f1)
    #     first_term = protocol.DeltaWd_single_hopping(numerical_estimation)
        
    #     fmin = numerical_estimation.fmin
    #     numerical_estimation.fmin = self.f1
    #     FTFprotocol = FTFProtocol(rF=self.rFprime, rU=self.rU)
    #     second_term = mp.exp(-numerical_estimation.model.kFtoU_mp(self.f1) / (self.rF * numerical_estimation.model.beta * numerical_estimation.model.xt)) * FTFprotocol.DeltaWd_single_hopping(numerical_estimation=numerical_estimation) #add M-1 to numerical_estimation
    #     numerical_estimation.fmin = fmin
        
    #     fmeanrF = (1/(numerical_estimation.model.beta*numerical_estimation.model.xt))*mp.log((numerical_estimation.model.beta*numerical_estimation.model.xt*self.rF)/numerical_estimation.model.k0)
    #     fmeanrFprime = (1/(numerical_estimation.model.beta*numerical_estimation.model.xt))*mp.log((numerical_estimation.model.beta*numerical_estimation.model.xt*self.rFprime)/numerical_estimation.model.k0)
    #     if self.f1 <= numerical_estimation.model.fc:
    #         third_term = numerical_estimation.model.xm*(fmeanrF-numerical_estimation.model.fc) - numerical_estimation.model.xm*(fmeanrFprime-numerical_estimation.model.fc)
    #     elif self.f1 > numerical_estimation.model.fc and self.f1 <= fmeanrFprime and self.f1 <= fmeanrF:
    #         third_term = numerical_estimation.model.xm*(fmeanrF-self.f1) - numerical_estimation.model.xm*(fmeanrFprime-self.f1)
    #     elif self.f1 > numerical_estimation.model.fc and self.f1 > fmeanrFprime and self.f1 <= fmeanrF:
    #         third_term = numerical_estimation.model.xm*(fmeanrF-self.f1)
    #     elif self.f1 > numerical_estimation.model.fc and self.f1 > fmeanrFprime and self.f1 > fmeanrF:
    #         third_term = 0
        
    #     numerical_estimation.model._cached_results[key] = first_term + second_term + third_term
    #     return float(numerical_estimation.model._cached_results[key])



    # def DeltaWd(self, outside_simulation, f:np.ndarray)->float:
    #     """returns the estimated dissipated work"""
    #     simulation = copy(outside_simulation)
        
    #     diffs = np.diff(f)
    #     bin_edges = np.concatenate([
    #     [f[0] - diffs[0] / 2],  # First edge
    #     f[:-1] + diffs / 2,  # Middle edges
    #     [f[-1] + diffs[-1] / 2]  # Last edge
    #     ])

    #     df= f[1] - f[0]
    #     fmin = simulation.fmin

    #     X,Y = np.meshgrid(f, f)
    #     PsF_list_rF = simulation.model.PsF(Y,X,self.rF)
    #     PsF_list_rFprime = simulation.model.PsF(Y,X,self.rFprime)
        
    #     simulation.N = 1000
    #     simulation.protocol = NOProtocol(self.rF)
    #     simulation.run()
    #     pU = np.mean(simulation.state_list, axis=1)
    #     bin_indices = np.digitize(simulation.force_list[:,0], bin_edges, right=True) #Return the indices of the bins to which each value in input array belongs.
    #     pU_force = np.zeros(len(f))
    #     for i in range(1, len(bin_edges)):
    #         pU_force[i-1] = np.mean(pU[bin_indices == i])
        
    #     simulation.N = 100

    #     Wd_list_rF = np.zeros_like(f)
    #     simulation.protocol = NOProtocol(self.rF)
    #     for k, f_ in np.ndenumerate(f):
    #         simulation.fmin = f_
    #         simulation.initial_state = True
    #         simulation.run(pre_thermalise=False)
    #         Wd_list_rF[k] = simulation.dissipated_work().mean()

    #     Wd_list_rU = np.zeros_like(f)
    #     simulation.protocol = NOProtocol(self.rU)
    #     for k, f_ in np.ndenumerate(f):
    #         simulation.fmin = f_
    #         simulation.initial_state = True
    #         simulation.run(pre_thermalise=False)
    #         Wd_list_rU[k] = simulation.dissipated_work().mean()

    #     simulation.reset()
    #     simulation.fmin = fmin
        
    #     DeltaWd = np.zeros_like(f)
    #     for k in range(len(f)):
    #         DeltaWd[k] = pU_force[k]*(Wd_list_rF[k] - Wd_list_rU[k]) + (1-pU_force[k])*((simulation.model.kFtoU(f[k:])*PsF_list_rF[k,k:]*Wd_list_rF[k:]).sum()*df/self.rF - (simulation.model.kFtoU(f[k:])*PsF_list_rFprime[k,k:]*Wd_list_rU[k:]).sum()*df/self.rFprime)

    #     return DeltaWd