import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random, normal
from hairpyn.dna_model import ControlledLengthModelWLC
from hairpyn.protocol.none import NOProtocol
from scipy.optimize import root_scalar

class ControlledLengthSimulation():
    def __init__(self, N:int=1, forward:bool=True, initial_state:bool=False, max_steps:int=1)->None:

        self.model = ControlledLengthModelWLC()

        self.xmin = 138 #nm
        self.xmax = 368 #nm
        self.Deltat = 1e-3 #s
        self.dx = 1e-3 #nm
        
        self.protocol = NOProtocol()
        
        self.N = N
        self.forward = forward
        self.initial_state = initial_state
        self.max_steps = max_steps

        self.fmax_int = 25 #pN
        self.fmin_int = 1e-5 #pN

    def run(self)->None:
        """
        Runs the simulation of the model.
        """

        if self.forward:
            self.initial_length = self.xmin
            self.final_length = self.xmax
        else:
            self.initial_length = self.xmax
            self.final_length = self.xmin
        
        self.initial_force = root_scalar(self.model.equationS1, bracket=[1e-5, self.fmax_int], args=(self.initial_state, self.initial_length)).root
        
        self.max_steps = self.protocol.controlled_length_max_steps(self.xmin, self.xmax, self.Deltat)
    
        self.length_list = np.zeros((self.max_steps, self.N), float)
        self.length_list_measured = np.zeros((self.max_steps, self.N), float)
        self.force_list = np.zeros((self.max_steps, self.N), float)
        self.force_list_measured = np.zeros((self.max_steps, self.N), float)
        self.time_list = np.zeros(self.max_steps, float)
        self.state_list = np.empty((self.max_steps, self.N), bool)
        self.rate_list = np.zeros((self.max_steps, self.N), float)
        self.mask = np.empty((self.max_steps, self.N), bool)
        
        self.length_list[0,:] = [self.initial_length]*self.N
        self.length_list_measured[0,:] = [self.initial_length]*self.N
        self.force_list[0,:] = [self.initial_force]*self.N
        self.force_list_measured[0,:] = [self.initial_force]*self.N
        self.time_list[0] = 0
        self.state_list[0,:] = [self.initial_state]*self.N
        self.rate_list[0,:] = [self.protocol.initial_rate]*self.N
        self.mask[0,:] = [False]*self.N
        
        for step in range(1, self.max_steps):
            
            self.time_list[step] = self.time_list[step-1] + self.Deltat
            finished_trajectories_mask = (self.rate_list[step-1]==0)

            self.length_list[step][~finished_trajectories_mask] = self.length_list[step-1][~finished_trajectories_mask] - (-1)**self.forward * self.rate_list[step-1][~finished_trajectories_mask] * self.Deltat / self.model.trapped_bead.kb

            unmasked_indices = np.where(~finished_trajectories_mask)[0]
            for idx in unmasked_indices:
                self.force_list[step,idx] = root_scalar(self.model.equationS1, bracket=[1e-5, self.fmax_int], args=(self.state_list[step-1,idx], self.length_list[step,idx])).root
            DNAhandles_length_Nlist = self.model.DNA_handles.length(self.state_list[step][~finished_trajectories_mask], self.force_list[step][~finished_trajectories_mask])
            DNAhairpin_length_Nlist = self.model.DNA_hairpin.length(self.state_list[step][~finished_trajectories_mask], self.force_list[step][~finished_trajectories_mask])

            kh_Nlist = (self.model.DNA_handles.force(self.state_list[step][~finished_trajectories_mask], DNAhandles_length_Nlist + self.dx) - self.force_list[step][~finished_trajectories_mask])/self.dx
            kDNA_Nlist = (self.model.DNA_hairpin.force(self.state_list[step][~finished_trajectories_mask], DNAhairpin_length_Nlist + self.dx) - self.force_list[step][~finished_trajectories_mask])/self.dx
            kmol_Nlist = (kh_Nlist*kDNA_Nlist)/(kh_Nlist+kDNA_Nlist)
            
            self.length_list_measured[step][~finished_trajectories_mask] = normal(self.length_list[step][~finished_trajectories_mask], 1/(self.model.beta*(kmol_Nlist+self.model.trapped_bead.kb)))
            self.force_list_measured[step][~finished_trajectories_mask] = normal(self.force_list[step][~finished_trajectories_mask], self.model.trapped_bead.kb**2/(self.model.beta*(kmol_Nlist+self.model.trapped_bead.kb)))

            rand_Nlist = random(self.N)

            for idx in unmasked_indices:
                if self.state_list[step-1, idx] == False:
                    if self.model.kFtoU(np.array([self.force_list[step, idx]]))[0] * self.Deltat > rand_Nlist[idx]:
                        self.state_list[step, idx] = True
                        self.mask[step, idx] = True
                    else:
                        self.state_list[step, idx] = False
                        self.mask[step, idx] = False
                elif self.state_list[step-1, idx] == True:
                    if self.model.kUtoF(np.array([self.force_list[step, idx]]))[0] * self.Deltat > rand_Nlist[idx]:
                        self.state_list[step, idx] = False
                        self.mask[step, idx] = True
                    else:
                        self.state_list[step, idx] = True
                        self.mask[step, idx] = False

            rates_to_update = (self.rate_list[step-1] != 0) & (self.length_list[step] >= self.xmin) & (self.length_list[step] <= self.xmax)
            self.rate_list[step][rates_to_update] = self.protocol.controlled_length_updated_rate(self.time_list[step], self.state_list[step][rates_to_update], self.rate_list[step-1][rates_to_update], self.length_list[step][rates_to_update])

            self.force_list[step][finished_trajectories_mask] = self.force_list[step-1][finished_trajectories_mask]
            self.length_list[step][finished_trajectories_mask] = self.length_list[step-1][finished_trajectories_mask]
            self.state_list[step][finished_trajectories_mask] = self.state_list[step-1][finished_trajectories_mask]

        measurement_times = self.time_list[np.argmin(self.rate_list, axis=0)]
        self.avg_time = (measurement_times.mean(), measurement_times.std())
        
        return None

    def show(self)->None:
        """
        Plots most important plot of a simulation
        """
        
        mask = self.rate_list != 0
        fig, axs = plt.subplots(3,1, figsize=(8,6))

        time_list = np.repeat(self.time_list[:, np.newaxis], self.N, axis=1)
        string = 'average measurement time = ' + str(np.round(self.avg_time[0], 2)) + ' s'

        axs[0].plot(time_list, self.state_list, c='tab:blue', label=string)
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
    
    def work(self, backward:bool=False)->tuple:
        """
        Returns the work in kBT for each simulation trajectory (after running the simulation).
        Makes no assumption on initial and final state (F or U)

        Returns:
        tuple: the average work and its standard deviation
        """
        finished_mask = self.rate_list != 0
        W_list = - np.sum(self.force_list * (np.roll(self.length_list, shift=-1, axis=0)-self.length_list)*np.roll(finished_mask, shift=-1, axis=0)*np.roll(finished_mask, shift=1, axis=0), axis=0)
        return (W_list.mean(), W_list.std(), W_list)
    
    def dissipated_work(self, backward:bool=False)->tuple:
        """
        Returns the dissipated work in kBT for each simulation trajectory (after running the simulation).
        Makes no assumption on initial and final state (F or U)

        Returns:
        tuple: the average dissipated work and its standard deviation
        """
        
        Wd_list = self.work()[-1] - self.model.DeltaGFU(self.force_list[-1]) + self.model.DeltaGFU(self.force_list[0])
        return (Wd_list.mean(), Wd_list.std(), Wd_list)