import matplotlib.pyplot as plt
import numpy as np
from hairpyn.rng import random
from hairpyn.dna_model import BEModel
from hairpyn.protocol.test import TestProtocol
from hairpyn.protocol.none import NOProtocol
from joblib import Parallel, delayed
from scipy.optimize import curve_fit, root_scalar
from scipy.integrate import quad
from copy import deepcopy

class Simulation():
    def __init__(self, N:int=1, forward:bool=True, initial_state:bool=False, measurement_error_rate:float=0, ensemble:str='LdF')->None:
        
        self.model = BEModel(ensemble)

        self._fmin = 8 #pN
        self._fmax = 22 #pN    
        
        self.Deltat = 1e-3 #s

        self.N = N
        self.initial_state = initial_state
        self.forward = forward
        self.measurement_error_rate = measurement_error_rate

        self._initial_force = self._fmin

        self._protocol = NOProtocol()

    @property
    def initial_force(self):
        return self._initial_force
    @initial_force.setter
    def initial_force(self, val:bool):
        self._initial_force = val
        self._initial_length = self.model.length(self.initial_state, val)
    
    @property
    def initial_length(self):
        return self._initial_length
    @initial_length.setter
    def initial_length(self, val:bool):
        self._initial_length = val
        self._initial_force = self.model.force(self.initial_state, val)

    @property
    def forward(self):
        return self._forward
    @forward.setter
    def forward(self, val:bool):
        self._forward = val
        if val:
            self.initial_force = self.fmin
            self.final_force = self.fmax
        else:
            self.initial_force = self.fmax
            self.final_force = self.fmin
    
    @property
    def fmin(self):
        return self._fmin
    @fmin.setter
    def fmin(self, val:float):
        self._fmin = val
        if self.forward:
            self.initial_force = val
        else:
            self.final_force = val
    
    @property
    def fmax(self):
        return self._fmax
    @fmax.setter
    def fmax(self, val:float):
        self._fmax = val
        if self.forward:
            self.final_force = val
        else:
            self.initial_force = val
    
    @property
    def protocol(self):
        return self._protocol
    @protocol.setter
    def protocol(self, val):
        self._protocol = val
        if val.name == 'MD' or val.name == 'CMD':
            self.initial_force = val.f0
            self.initial_length = self.model.length(self.initial_state, val.f0)

    def thermalise(self,  f:float, thermalisation_time:float)->None:
        """
        Runs the thermalisation of the model.
        """

        simulation = Simulation(self.N)
        simulation.model = self.model
        simulation.protocol = NOProtocol(0, thermalisation_time)
        simulation.initial_force = f
        simulation.run(pre_thermalise=False)
        return simulation.state_list[-1]

    def reset(self)->None:
        """
        Fully releases memory used by the simulation by setting data structures to None.
        """

        self.observation_index = None
        self.binary_history = None

        self.force_list = None
        self.length_list = None
        self.time_list = None
        self.state_list = None
        self.rate_list = None

        self.force_list_backward = None
        self.length_list_backward = None
        self.time_list_backward = None
        self.state_list_backward = None
        self.rate_list_backward = None

        return None

    def run(self, pre_thermalise:bool=True)->None:
        """
        Runs the simulation of the model.
        """
        
        self.observation_index = np.zeros(self.N, dtype=int)
        self.binary_history = np.zeros(self.N, dtype=int)
        max_steps = self.protocol.max_steps(self.model, self.fmin, self.fmax, self.Deltat)
        self.force_list = np.zeros((max_steps, self.N), dtype=float)
        self.length_list = np.zeros((max_steps, self.N), dtype=float)
        self.time_list = np.zeros(max_steps, dtype=float)
        self.state_list = np.empty((max_steps, self.N), dtype=bool) #False <-> Folded
        self.rate_list = np.zeros((max_steps, self.N), dtype=float)
        
        ### INITIALIZATION ###
        self.force_list[0] = [self.initial_force]*self.N
        self.length_list[0] = [self.initial_length]*self.N
        self.state_list[0] = self.thermalise(self.initial_force, 1) if pre_thermalise else [self.initial_state]*self.N
        self.rate_list[0] = [self.protocol.initial_rate]*self.N
        
        if self.model.ensemble == 'LdF':
            def cv_update(state:np.ndarray, force:np.ndarray, length:np.ndarray, rate:np.ndarray):
                f = force - (-1)**self.forward * rate*self.Deltat
                l = self.model.length(state, f) #wrong, not the right ensemble
                return f, l
        elif self.model.ensemble == 'FdL':
            def cv_update(state:np.ndarray, force:np.ndarray, length:np.ndarray, rate:np.ndarray):
                l = length - (-1)**self.forward * rate/self.model.kx * self.Deltat
                f = self.model.force(state, l)
                return f, l

        ### MONTE CARLO ###
        measured_states = np.zeros(self.N, bool)
        for step in range(1, max_steps):
            self.time_list[step] = self.time_list[step-1] + self.Deltat
            self.force_list[step], self.length_list[step] = cv_update(self.state_list[step-1], self.force_list[step-1],  self.length_list[step-1], self.rate_list[step-1])

            unchanged_mask = np.ones(self.N, dtype=bool) if step==1 else (self.rate_list[step-1]==self.rate_list[step-2]) 
            
            random_list = random(self.N)
            
            FtoUtransition_mask = np.zeros(self.N, dtype=bool)
            UtoFtransition_mask = np.zeros(self.N, dtype=bool)

            FtoUtransition_mask[(~self.state_list[step-1])&unchanged_mask] = (self.model.kFtoU(self.force_list[step-1][(~self.state_list[step-1])&unchanged_mask]) * self.Deltat > random_list[(~self.state_list[step-1])&unchanged_mask])
            UtoFtransition_mask[self.state_list[step-1]&unchanged_mask] = (self.model.kUtoF(self.force_list[step-1][self.state_list[step-1]&unchanged_mask]) * self.Deltat > random_list[self.state_list[step-1]&unchanged_mask])
            transition_mask = FtoUtransition_mask + UtoFtransition_mask

            self.state_list[step][~transition_mask] = self.state_list[step-1][~transition_mask]
            self.state_list[step][transition_mask] = ~self.state_list[step-1][transition_mask]

            if self.protocol.name == 'MD' or self.protocol.name =='CMD':
                finished_trajectories = np.isnan(self.rate_list[step-1])
            else:
                finished_trajectories = (np.isnan(self.rate_list[step-1])) | (self.force_list[step] <= self.fmin) | (self.force_list[step] >= self.fmax)

            error_mask = (random(self.N) <= self.measurement_error_rate)
            measured_states[error_mask] = ~self.state_list[step][error_mask]
            measured_states[~error_mask] = self.state_list[step][~error_mask]
            
            results = self.protocol.updated_rate(self.observation_index[~finished_trajectories], self.binary_history[~finished_trajectories], measured_states[~finished_trajectories], self.rate_list[step-1][~finished_trajectories], self.force_list[step][~finished_trajectories], self.length_list[step][~finished_trajectories], forward=self.forward)
            
            self.observation_index[~finished_trajectories] = results['obs_idx']
            self.binary_history[~finished_trajectories] = results['binary_hist']
            
            self.rate_list[step][~finished_trajectories] = results['rate']
            self.rate_list[step][finished_trajectories] = np.full_like(self.rate_list[step][finished_trajectories], np.nan)

        return None

    def avg_time(self):
        t = self.time_list[np.argmax(np.isnan(self.rate_list), axis=0)] - self.time_list[np.argmin(np.isnan(self.rate_list), axis=0)]
        return t.mean(), t.std()
    
    def run_backward(self)->None:
        """
        Runs the simulation of return trip of the model.
        """
        
        self.force_list_backward = self.force_list[::-1, :]
        self.length_list_backward = self.length_list[::-1, :]
        self.time_list_backward = self.time_list
        self.state_list_backward = np.empty_like(self.state_list, dtype=bool) #False <-> Folded
        self.rate_list_backward = self.rate_list[::-1, :]
        
        self.state_list_backward[0] = self.state_list[-1]

        for step in range(1, self.state_list.shape[0]):
            
            unchanged_mask = (self.rate_list_backward[step]==self.rate_list_backward[step-1])

            random_list = random(self.N)

            FtoUtransition_mask = np.zeros(self.N, dtype=bool)
            UtoFtransition_mask = np.zeros(self.N, dtype=bool)
        
            FtoUtransition_mask[(~self.state_list_backward[step-1])&unchanged_mask] = (self.model.kFtoU(self.force_list_backward[step-1][(~self.state_list_backward[step-1])&unchanged_mask]) * self.Deltat > random_list[(~self.state_list_backward[step-1])&unchanged_mask])
            UtoFtransition_mask[self.state_list_backward[step-1]&unchanged_mask] = (self.model.kUtoF(self.force_list_backward[step-1][self.state_list_backward[step-1]&unchanged_mask]) * self.Deltat > random_list[self.state_list_backward[step-1]&unchanged_mask])

            transition_mask = FtoUtransition_mask + UtoFtransition_mask

            self.state_list_backward[step][~transition_mask] = self.state_list_backward[step-1][~transition_mask]
            self.state_list_backward[step][transition_mask] = ~self.state_list_backward[step-1][transition_mask]

    def work(self, backward:bool=False)->tuple:
        """
        Returns the work in kBT for each simulation trajectory (after running the simulation).
        Makes no assumption on initial and final state (F or U).

        Returns:
        ndarray: the average work and its standard deviation for each trajectory in kBT units.
        """
        if backward:
            Wf_list = - self.model.xm * np.nansum(self.state_list_backward*(-self.rate_list_backward)*self.Deltat, axis=0)*( - (-1)**self.forward)
        else:
            Wf_list = - self.model.xm * np.nansum(self.state_list*self.rate_list*self.Deltat, axis=0)*( - (-1)**self.forward)
        
        if self.model.ensemble=='LdF':
            return self.model.beta*Wf_list
        elif self.model.ensemble=='FdL':
            if backward:
                index_first_non_nan = np.argmin(np.isnan(self.rate_list_backward), axis=0)
                index_last_non_nan = np.argmax(np.isnan(self.rate_list_backward), axis=0)
                index_last_non_nan[index_last_non_nan==0] = -1
                Wl_list = Wf_list + (self.force_list_backward[index_last_non_nan,np.arange(self.N)]*self.length_list_backward[index_last_non_nan,np.arange(self.N)] - self.force_list_backward[index_first_non_nan, np.arange(self.N)]*self.length_list_backward[index_first_non_nan, np.arange(self.N)] - 1/(2*self.model.kx) * (self.force_list_backward[index_last_non_nan,np.arange(self.N)]**2 - self.force_list_backward[index_first_non_nan, np.arange(self.N)]**2))
            else:
                index_first_non_nan = np.argmin(np.isnan(self.rate_list), axis=0)
                index_last_non_nan = np.argmax(np.isnan(self.rate_list), axis=0)
                Wl_list = Wf_list + (self.force_list[index_last_non_nan,np.arange(self.N)]*self.length_list[index_last_non_nan,np.arange(self.N)] - self.force_list[index_first_non_nan, np.arange(self.N)]*self.length_list[index_first_non_nan, np.arange(self.N)] - 1/(2*self.model.kx) * (self.force_list[index_last_non_nan,np.arange(self.N)]**2 - self.force_list[index_first_non_nan, np.arange(self.N)]**2))
            return self.model.beta*Wl_list
    
    def dissipated_work(self, backward:bool=False)->tuple:
        """
        Returns the dissipated work in kBT for each simulation trajectory (after running the simulation).
        Makes no assumption on initial and final state (F or U).

        Returns:
        ndarray: the average dissipated work and its standard deviation for each trajectory in kBT units.
        """
        if backward:
            index_first_non_nan = np.argmin(np.isnan(self.rate_list_backward), axis=0)
            index_last_non_nan = np.argmax(np.isnan(self.rate_list_backward), axis=0)
            index_last_non_nan[index_last_non_nan==0] = -1
            Wd_list = self.work(backward=backward) - self.model.beta*(self.model.DeltaG(self.state_list_backward[index_last_non_nan,np.arange(self.N)], self.force_list_backward[index_last_non_nan,np.arange(self.N)])- self.model.DeltaG(self.state_list_backward[index_first_non_nan,np.arange(self.N)], self.force_list_backward[index_first_non_nan,np.arange(self.N)]))
        else:
            index_first_non_nan = np.argmin(np.isnan(self.rate_list), axis=0)
            index_last_non_nan = np.argmax(np.isnan(self.rate_list), axis=0)
            Wd_list = self.work(backward=backward) - self.model.beta*(self.model.DeltaG(self.state_list[index_last_non_nan, np.arange(self.N)], self.force_list[index_last_non_nan, np.arange(self.N)]) - self.model.DeltaG(self.state_list[index_first_non_nan, np.arange(self.N)], self.force_list[index_first_non_nan, np.arange(self.N)]))
        return Wd_list
    
    def show(self, backward=False)->None:
        """
        Plots most important plot of a simulation
        """
        
        fig, axs = plt.subplots(2,2, figsize=(8,5))
        

        if backward:
            axs[0,0].plot(self.time_list_backward, self.force_list_backward, c='tab:blue')
            axs[0,0].set_ylabel(r'Force [pN]', fontsize=14)
            axs[0,0].legend()

            axs[1,0].plot(self.time_list_backward, self.length_list_backward, c='tab:blue')
            axs[1,0].set_ylabel(r'Length [nm]', fontsize=14)
            axs[1,0].set_xlabel(r'Time [s]', fontsize=14)

            axs[0,1].plot(self.time_list_backward, self.rate_list_backward, c='tab:blue')
            axs[0,1].set_ylabel(r'Rate [pN/s]', fontsize=14)
            axs[0,1].set_xlabel(r'Time [s]', fontsize=14)

            axs[1,1].plot(self.length_list_backward, self.force_list_backward, c='tab:blue')
            axs[1,1].invert_xaxis()
            axs[1,1].set_ylabel(r'Force [pN]', fontsize=14)
            axs[1,1].set_xlabel(r'Length [nm]', fontsize=14)
        else:
            axs[0,0].plot(self.time_list, self.force_list, c='tab:blue')
            axs[0,0].set_ylabel(r'Force [pN]', fontsize=14)
            axs[0,0].legend()

            axs[1,0].plot(self.time_list, self.length_list, c='tab:blue')
            axs[1,0].set_ylabel(r'Length [nm]', fontsize=14)
            axs[1,0].set_xlabel(r'Time [s]', fontsize=14)

            axs[0,1].plot(self.time_list, self.rate_list, c='tab:blue')
            axs[0,1].set_ylabel(r'Rate [pN/s]', fontsize=14)
            axs[0,1].set_xlabel(r'Time [s]', fontsize=14)

            axs[1,1].plot(self.length_list, self.force_list, c='tab:blue')
            axs[1,1].set_ylabel(r'Force [pN]', fontsize=14)
            axs[1,1].set_xlabel(r'Length [nm]', fontsize=14)

        fig.tight_layout()

        return None

    def curve_func(self, x, Upsilon):
        return x+Upsilon
    
    def Upsilon(self, Wd_forward:np.ndarray, Wd_backward:np.ndarray, equation:str='Crooks', show:bool=False)->tuple:
        """
        Takes Wd_forward and Wd_backward and type: 'Crooks', 'C', 'Jarzynski', 'J', 'Bennett', 'B'. Returns an estimation for Upsilon.
        Parameters:
        - Wd_forward (np.ndarray): the forward dissipated work.
        - Wd_backward (np.ndarray): the backward dissipated work.
        - type (str): the equality used for the computation.

        Returns:
        tuple: The estimated Upsilon and its standard deviation
        """
        
        if len(Wd_forward)==0 or len(Wd_backward)==0:
            print('Missing work trajectories')
            return np.nan, np.nan
        
        if show:
            fig, axs = plt.subplots(1,2, figsize=(8,5))
            
            axs[0].hist(Wd_forward, bins=100, density=True, alpha=.7, label=r'$\rho_\rightarrow(W_d)$')
            axs[0].hist(-Wd_backward, bins=100, density=True, alpha=.7, label=r'$\rho_\leftarrow(-W_d)$')
            axs[0].set_xlabel(r'$W_d$ [$k_B T$]', fontsize=15)
            axs[0].set_ylabel(r'$\rho(W_d)$', fontsize=15)
            axs[0].legend(fontsize=14)

            bins = np.histogram_bin_edges(np.concatenate((Wd_forward, -Wd_backward)), bins=100)
            rho_forward, _ = np.histogram(Wd_forward, bins=bins, density=True)
            rho_backward, _ = np.histogram(-Wd_backward, bins=bins, density=True)
            Wd = ((bins[:-1] + bins[1:]) / 2)

            mask = (rho_backward > 1e-15)&(rho_forward > 1e-15)
            xdata = Wd[mask]
            ydata = np.log(rho_forward[mask]/rho_backward[mask])

            axs[1].scatter(xdata, ydata)
            axs[1].set_xlabel(r'$W_d$ [$k_B T$]', fontsize=15)
            axs[1].set_ylabel(r'$\log \left(\frac{\rho_\rightarrow(W_d)}{\rho_\leftarrow(-W_d)}\right)$', fontsize=15)
            axs[1].grid()

            fig.tight_layout()

        if equation == 'Crooks' or equation == 'C':
            bins = np.histogram_bin_edges(np.concatenate((Wd_forward, -Wd_backward)), bins=100)
            rho_forward, _ = np.histogram(Wd_forward, bins=bins, density=True)
            rho_backward, _ = np.histogram(-Wd_backward, bins=bins, density=True)
            Wd = ((bins[:-1] + bins[1:]) / 2)

            mask = (rho_backward > 1e-15)&(rho_forward > 1e-15)
            xdata = Wd[mask]
            ydata = np.log(rho_forward[mask]/rho_backward[mask])
            if len(ydata) == 0:
                Upsilon = np.nan
                Upsilon_std = np.nan
            else:
                popt, pcov = curve_fit(self.curve_func, xdata, ydata, p0=[0])
                Upsilon = float(popt[0])
                Upsilon_std = float(np.sqrt(pcov[0, 0]))
            
        
        elif equation == 'Jarzynski' or equation == 'J':
            bins = np.histogram_bin_edges(Wd_forward, bins=100)
            rho_forward, _ = np.histogram(Wd_forward, bins=bins, density=True)
            Wd = ((bins[:-1] + bins[1:]) / 2)
            DeltaWd =  bins[1:] - bins[:-1]
            Upsilon = np.log((np.exp(-Wd)*rho_forward*DeltaWd).sum())
            Upsilon_std = None
        
        elif equation == 'Bennett' or equation == 'B':
            nU = len(Wd_forward)
            nF = len(Wd_backward)
            def z_f(x):
                return np.log(1/nU * (np.exp(-Wd_forward)/(1+nU/nF * np.exp(-Wd_forward+x))).sum())
            def z_b(x):
                return np.log(1/nF * (1/(1+nU/nF * np.exp(Wd_backward+x))).sum())
            def to_zero_0(x):
                return (z_f(x) - z_b(x))-x
            Upsilon = root_scalar(to_zero_0, bracket=[-100, 100], xtol=1e-7).root
            Upsilon_std = None
        
        return Upsilon, Upsilon_std

    def Wd0_estimation(self)->float:
        """
        Takes the considered protocol, returns Wd0 (the dissipated work without feedback) in units of kBT
        and its associated type B standard deviation, in the mean-field approximation (Eq. 25 of the paper).
        Assumes we start at full F and end at full U.

        Returns:
        float: Wd0 (the dissipated work without feedback).
        float: sigma(Wd0) (the type B estimated numerical error)
        """

        if self.model.ensemble=='LdF':
            mu = 2*self.model.xt/self.model.xm - 1
            rtilde = self.model.beta*self.model.xm*self.protocol.initial_rate/(4*self.model.kFtoU(self.model.fc))

            def integrand1(z):
                return np.exp(mu*z)*np.cosh(z)
            def integrand2(y, x):
                result, error = quad(integrand1, y, x)
                return 1/np.cosh(y)**2 * np.exp(-1/rtilde * result)
            def integrand3(x):
                result, error =  quad(integrand2, -10, x, args=(x))
                return result

            return quad(integrand3, -10, 10)
        elif self.model.ensemble=='FdL':
            return None
    
    def Wd0(self, N:int=5000)->float:
        """
        Takes the considered protocol, returns Wd0 (the dissipated work without feedback) in units of kBT
        and its associated type A standard deviation.
        Computes the mean dissipated work on N=5000 trajectories.

        Returns:
        float: Wd0 (the dissipated work without feedback).
        float: sigma(Wd0) (the type A standard deviation of the estimation)
        """

        simulation = Simulation(N=N, forward=self.forward, initial_state=self.initial_state)
        simulation.model = self.model
        simulation.protocol = NOProtocol(rF=self.protocol.initial_rate)
        simulation.run()

        return simulation.dissipated_work().mean(), simulation.dissipated_work().std()