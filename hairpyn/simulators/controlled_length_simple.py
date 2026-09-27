import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random, normal
from hairpyn.dna_model import ControlledLengthModel
from hairpyn.protocol.none import NOProtocol
from scipy.optimize import curve_fit

class ControlledLengthSimpleSimulation():
    def __init__(self, N:int=1, forward:bool=True, initial_state:bool=False, max_steps:int=1)->None:

        self.model = ControlledLengthModel()

        self.fmin = 8 #pN
        self.fmax = 22 #pN
        self.Deltat = 1e-3 #s
        self.dx = 1e-3 #nm
        
        self.protocol = NOProtocol()
        
        self.N = N
        self.forward = forward
        self.initial_state = initial_state
        self.max_steps = max_steps

    def run(self)->None:
        """
        Runs the simulation of the model.
        """

        if self.forward:
            self.initial_force = self.fmin
            self.initial_length = self.model.length(self.initial_state, self.fmin)
            self.final_force = self.fmax
            self.final_length = self.model.length(True, self.fmax)
        else:
            self.initial_force = self.fmax
            self.initial_force = self.model.length(self.initial_state, self.fmax)
            self.final_force = self.fmin
            self.final_force = self.model.length(False, self.fmin)
        
        #self.max_steps = self.protocol.controlled_length_max_steps(self.fmin, self.fmax, self.Deltat)
        self.max_steps = self.protocol.controlled_force_max_steps(self.fmin, self.fmax, self.Deltat)
        self.max_steps +=1000

        self.force_list = np.zeros((self.max_steps, self.N), float)
        self.length_list = np.zeros((self.max_steps, self.N), float)
        self.time_list = np.zeros(self.max_steps, float)
        self.state_list = np.empty((self.max_steps, self.N), bool) #False <-> Folded
        self.rate_list = np.zeros((self.max_steps, self.N), float)
        self.FtoUtransition_mask = np.full((self.max_steps, self.N), False)
        self.UtoFtransition_mask = np.full((self.max_steps, self.N), False)
        self.transition_mask = np.full((self.max_steps, self.N), False)
        
        self.force_list[0] = [self.initial_force]*self.N
        self.length_list[0] = [self.initial_length]*self.N
        self.state_list[0] = [self.initial_state]*self.N
        self.rate_list[0] = [self.protocol.initial_rate]*self.N

        for step in range(1, self.max_steps):
            self.time_list[step] = self.time_list[step-1] + self.Deltat
            self.length_list[step] = self.length_list[step-1] - (-1)**self.forward * self.rate_list[step-1]/self.model.kx * self.Deltat
            self.force_list[step] = self.model.force(self.state_list[step-1], self.length_list[step])

            random_list = random(self.N)

            self.FtoUtransition_mask[step][~self.state_list[step-1]] = (self.model.kFtoU(self.length_list[step-1][~self.state_list[step-1]]) * self.Deltat > random_list[~self.state_list[step-1]])
            self.UtoFtransition_mask[step][self.state_list[step-1]] = (self.model.kUtoF(self.length_list[step-1][self.state_list[step-1]]) * self.Deltat > random_list[self.state_list[step-1]])

            self.transition_mask[step] = self.FtoUtransition_mask[step] + self.UtoFtransition_mask[step]

            self.state_list[step] = self.state_list[step-1]
            self.state_list[step][self.transition_mask[step]] = ~self.state_list[step-1][self.transition_mask[step]]

            rates_to_update = (self.rate_list[step-1] != 0) & (self.force_list[step] >= self.fmin) & (self.force_list[step] <= self.fmax)
            self.rate_list[step][rates_to_update] = self.protocol.controlled_length_updated_rate(self.time_list[step], self.state_list[step][rates_to_update], self.rate_list[step-1][rates_to_update], self.length_list[step][rates_to_update])
        
        return None

    def run_backward(self):
        """
        Runs the simulation of return trip of the model.
        """

        if self.forward:
            self.initial_force = self.fmax
            self.initial_length = self.model.length(self.initial_state, self.fmax)
            self.final_force = self.fmax
            self.final_length = self.model.length(True, self.fmin)
        else:
            self.initial_force = self.fmin
            self.initial_force = self.model.length(self.initial_state, self.fmin)
            self.final_force = self.fmax
            self.final_force = self.model.length(False, self.fmax)
        
        self.length_list_backward = self.length_list[::-1, :]
        self.time_list_backward = self.time_list
        self.state_list_backward = np.empty((self.max_steps, self.N), bool) #False <-> Folded
        self.force_list_backward = np.zeros((self.max_steps, self.N))
        self.rate_list_backward = self.rate_list[::-1, :]
        self.FtoUtransition_mask_backward = np.full((self.max_steps, self.N), False)
        self.UtoFtransition_mask_backward = np.full((self.max_steps, self.N), False)
        self.transition_mask_backward = np.full((self.max_steps, self.N), False)
        
        self.state_list_backward[0] = self.state_list[-1]

        for step in range(1, self.max_steps):

            random_list = random(self.N)

            self.FtoUtransition_mask_backward[step][~self.state_list_backward[step-1]] = (self.model.kFtoU(self.length_list_backward[step-1][~self.state_list_backward[step-1]]) * self.Deltat > random_list[~self.state_list_backward[step-1]])
            self.UtoFtransition_mask_backward[step][self.state_list_backward[step-1]] = (self.model.kUtoF(self.length_list_backward[step-1][self.state_list_backward[step-1]]) * self.Deltat > random_list[self.state_list_backward[step-1]])

            self.transition_mask_backward[step] = self.FtoUtransition_mask_backward[step] + self.UtoFtransition_mask_backward[step]

            self.state_list_backward[step] = self.state_list_backward[step-1]
            self.state_list_backward[step][self.transition_mask_backward[step]] = ~self.state_list_backward[step-1][self.transition_mask_backward[step]]
            
            self.force_list_backward[step] = self.model.force(self.state_list[step], self.length_list[step])
    
    def avg_time(self):
        measurement_times = self.time_list[np.argmin(self.rate_list, axis=0)]
        return measurement_times
    
    def show(self)->None:
        """
        Plots most important plot of a simulation
        """
        
        mask = self.rate_list != 0
        fig, axs = plt.subplots(3,1, figsize=(8,6))

        time_list = np.repeat(self.time_list[:, np.newaxis], self.N, axis=1)

        axs[0].plot(time_list, self.state_list, c='tab:blue')
        axs[0].set_ylabel('State', fontsize=14)
        axs[0].set_xlabel('Time [s]', fontsize=14)
        axs[0].legend()

        axs[1].plot(self.length_list[mask], self.force_list[mask], c='tab:blue')
        axs[1].set_ylabel('Force [pN]', fontsize=14)
        axs[1].set_xlabel('Length [nm]', fontsize=14)

        axs[2].plot(time_list, self.rate_list, c='tab:blue')
        axs[2].set_ylabel('Rate [pN/s]', fontsize=14)
        axs[2].set_xlabel('Time [s]', fontsize=14)

        if self.protocol.name == 'DTF' or self.protocol.name == 'strategy':
            axs[1].axvline(self.protocol.lambda1, c='tab:orange')

        fig.tight_layout()

        return None

    def Wf(self, backward:bool=False):
        if backward:
            return - self.model.beta * self.model.xm * np.sum(self.state_list_backward*(-self.rate_list_backward)*self.Deltat, axis=0)
        else:
            return - self.model.beta * self.model.xm * np.sum(self.state_list*self.rate_list*self.Deltat, axis=0)
    
    def Wlambda(self, backward:bool=False):
        if backward:
            return self.Wf(backward) + self.model.beta * (self.force_list_backward[-1]*self.length_list_backward[-1] - self.force_list_backward[0]*self.length_list_backward[0] - 1/(2*self.model.kx) * (self.force_list_backward[-1]**2 - self.force_list_backward[0]**2))
        else:
            return self.Wf(backward) + self.model.beta * (self.force_list[-1]*self.length_list[-1] - self.force_list[0]*self.length_list[0] - 1/(2*self.model.kx) * (self.force_list[-1]**2 - self.force_list[0]**2))
    
    def dissipated_work(self, backward:bool=False):
        if backward:
            return self.Wlambda(backward) - self.model.beta * (self.state_list_backward[-1]*(self.model.DeltaG0 - self.force_list_backward[-1]*self.model.xm) - self.state_list_backward[0]*(self.model.DeltaG0 - self.force_list_backward[0]*self.model.xm))
        else:
            return self.Wlambda(backward) - self.model.beta * (self.state_list[-1]*(self.model.DeltaG0 - self.force_list[-1]*self.model.xm) - self.state_list[0]*(self.model.DeltaG0 - self.force_list[0]*self.model.xm))


    def curve_func(self, x, gamma):
        return x+gamma

    def gamma_computation(self, iterator:float, plot:bool=False):
        """
        Takes the measuring force lambda1 or the pulling ratio rU/rF, returns an estimation for gamma and a plot if plot==True.
    
        Parameters:
        - iterator (float): the measuring force 'f1' or the pulling ratio 'ratio'
        - plot (bool): The plot option (True for a plot to be produced)

        Returns:
        float: The estimated gamma
        """
        if self.protocol.name == 'DTF' or self.protocol.name == 'strategy':
            self.protocol.lambda1 = iterator
        if self.protocol.name == 'CTF':
            self.protocol.rU = self.protocol.rF * iterator
        
        self.run()
        Wd_forward = self.dissipated_work()
        self.run_backward()
        Wd_backward = self.dissipated_work(backward=True)

        bins = np.histogram_bin_edges(np.concatenate((Wd_forward, Wd_backward)), bins=100)
        rho_forward, _ = np.histogram(Wd_forward, bins=bins, density=True)
        rho_backward, _ = np.histogram(-Wd_backward, bins=bins, density=True)
        Wd = ((bins[:-1] + bins[1:]) / 2)

        # mask0 = (rho_backward > 1e-5)&(rho_forward > 1e-5) #np.full_like(Wd, True, dtype=bool)
        # xdata = Wd[mask0]
        # ydata = np.log(rho_forward[mask0]/rho_backward[mask0])

        # mask = np.full_like(xdata, True, dtype=bool) #(-2 <= xdata) & (xdata <= 2)
        # popt, pcov = curve_fit(self.curve_func, xdata[mask], ydata[mask], p0=[0])
        # gamma = float(popt)
        # gamma_std = float(pcov)

        if plot:

            fig, axs = plt.subplots(1,2, figsize=(8,4))

            axs[0].hist(Wd_forward, bins, alpha=.8, label=r'$\rho_\rightarrow(W_d)$', density=True)
            axs[0].hist(-Wd_backward, bins, alpha=.8, label=r'$\rho_\leftarrow(-W_d)$', density=True)
            axs[0].legend(fontsize=12)
            axs[0].set_xlabel(r'$W_d$ [$k_B T$]', fontsize=15)
            axs[0].set_ylabel(r'$\rho_(W)$', fontsize=15)

            axs[1].scatter(xdata[~mask], ydata[~mask], c='tab:blue')
            axs[1].scatter(xdata[mask], ydata[mask], c='tab:orange')
            # axs[1].plot(xdata, self.curve_func(xdata, gamma), c='tab:purple', label=r"$\Upsilon$ = {:.3f} $\pm$ {:.3f}".format(gamma, gamma_std))
            # axs[1].plot(xdata, self.curve_func(xdata, gamma+gamma_std), c='tab:purple', ls='dotted')
            # axs[1].plot(xdata, self.curve_func(xdata, gamma-gamma_std), c='tab:purple', ls='dotted')

            axs[1].set_xlabel(r'$W_d$ [$k_B T$]', fontsize=15)
            axs[1].set_ylabel(r'$\ln \left( \frac{\rho_\rightarrow(W_d)}{\rho_\leftarrow(-W_d)} \right)$', fontsize=15)
            axs[1].grid()
            axs[1].legend(fontsize=15)

            axs[1].axvline(-2, c='tab:purple', ls=':')
            axs[1].axvline(2, c='tab:purple', ls=':')

            fig.suptitle(self.protocol.name, fontsize=20)

            fig.tight_layout()

        return gamma, gamma_std, Wd_forward.mean(), Wd_forward.std()