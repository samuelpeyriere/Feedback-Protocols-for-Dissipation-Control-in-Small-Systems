import matplotlib.pyplot as plt
import numpy as np


class Protocol():
    
    def __init__(self):

        self._observations = np.array([])
        self._rate_mask = np.array([])
        self._initial_rate = 0

        self.name = 'built protocol'
        self.dtype = [('rate', 'f8'), ('obs_idx', 'i4'), ('binary_hist', 'i4')]
    
    @property
    def initial_rate(self):
        return self._initial_rate
    @initial_rate.setter
    def initial_rate(self, val):
        self._initial_rate = val
        self._rate_mask[0][0] = val
    
    @property
    def rate_mask(self):
        return self._rate_mask
    @rate_mask.setter
    def rate_mask(self, val):
        self._rate_mask = np.array([np.array(elem, dtype=float) for elem in val], dtype=object)
        self._initial_rate = self._rate_mask[0][0]

    @property
    def observations(self):
        return self._observations
    @observations.setter
    def observations(self, val):
        observations = []
        for elem in val:
            if isinstance(elem, list):
                observations.append([float(subelem) for subelem in elem])
            else:
                observations.append(float(elem))
        self._observations = observations
        observations = None
    
    def show(self):
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        for k, elem in enumerate(self.observations):

            ys = np.linspace(-2*(1-2**(-k-1)), 2*(1-2**(-k-1)), 2**(k+1))
                
            if type(elem)==list: #CTF
                x = np.linspace(elem[0], elem[1], 2)
                for i, y in enumerate(ys):
                    #if self.rate_mask[k][i]
                    if i%2:
                        color = 'tab:blue'
                    else:
                        color = 'tab:orange'
                    ax.plot(x, np.full_like(x, y), c=color)
                ax.plot(x, 0*x, c='tab:gray')
            else:
                for i, y in enumerate(ys):
                    if i%2:
                        color = 'tab:blue'
                    else:
                        color = 'tab:orange'
                    ax.scatter(elem, y, c=color)
                ax.scatter(elem, 0, c='tab:gray')
        
        ax.set_xlim(8,22)
        ax.set_xlabel(r'Measurement Force [pN]', fontsize = 14)
        ax.set_ylabel(r'Measurement Step', fontsize = 14)
        ax.set_yticks([])
        
        fig.suptitle(r'Protocol $\rightarrow$', fontsize = 20)
        fig.tight_layout()
        

    def controlled_force_updated_rate(self, observation_index:np.ndarray, binary_history:np.ndarray, state_list:np.ndarray, previous_rate_list:np.ndarray, force_list:np.ndarray, forward:bool=True)->np.ndarray:
        """
        Takes the time, states, previous rates and forces of the system, returns the updated pull rates according to the DTF+CTF strategy.
    
        Parameters:
        - pobservation_index (ndarray): the index of the observation event
        - binary_history (ndarray): holds the information of all previous observations
        - state_list (ndarray): The hairpin's states (Folded=False or Unfolded=True)
        - previous_rate_list (ndarray): The previous pulling rates (pN/s)
        - force_list (ndarray): The forces pulling on the systems (pN)
        - forward (bool): The pull direction on the system

        Returns:
        ndarray: The updated pull rates
        """
        to_return = np.empty_like(previous_rate_list, dtype=self.dtype)
        if len(observation_index) == 0:
            return to_return
        else:
            finished_observations = (observation_index == len(self.observations))
            updated_rate = np.empty_like(previous_rate_list, dtype='float')
            updated_rate[finished_observations] = previous_rate_list[finished_observations]

            mem_obs = observation_index[:]
            mem_bin = binary_history[:]

            if np.any(~finished_observations):
                observation_index = observation_index[~finished_observations]
                binary_history = binary_history[~finished_observations]
                state_list = state_list[~finished_observations]
                previous_rate_list = previous_rate_list[~finished_observations]
                force_list = force_list[~finished_observations]
                
                observation_force_list = np.array([self.observations[idx] for idx in observation_index])
                rate_if_update_list = np.array([self.rate_mask[elem+1][2 * binary_history[k] + state_list[k]] for k, elem in np.ndenumerate(observation_index)])

                continuous_mask = ~np.array([isinstance(elem, float) for elem in observation_force_list])
                mask = np.zeros_like(force_list, dtype=bool) #True if a change to the rate must be made
                if continuous_mask.any():
                    mask[continuous_mask] = (force_list[continuous_mask] > np.array([item[0] for item in observation_force_list[continuous_mask]])) & (force_list[continuous_mask] < np.array([item[1] for item in observation_force_list[continuous_mask]]))
                    mask_end = (force_list[continuous_mask] >= np.array([item[1] for item in observation_force_list[continuous_mask]]))
                    observation_index[continuous_mask] = np.where(mask_end | (mask[continuous_mask]&(~np.isnan(rate_if_update_list[continuous_mask]))), observation_index[continuous_mask]+1, observation_index[continuous_mask])
                    binary_history[continuous_mask] = np.where(mask_end | (mask[continuous_mask]&(~np.isnan(rate_if_update_list[continuous_mask]))), 2 * binary_history[continuous_mask] + state_list[continuous_mask], binary_history[continuous_mask])
                    
                if (~continuous_mask).any():
                    mask[~continuous_mask] = (observation_force_list[~continuous_mask] - self.half_interval_force < force_list[~continuous_mask]) & (force_list[~continuous_mask] <= observation_force_list[~continuous_mask] + self.half_interval_force)
                    observation_index[~continuous_mask] = np.where(mask[~continuous_mask], observation_index[~continuous_mask]+1, observation_index[~continuous_mask])
                    binary_history[~continuous_mask] = np.where(mask[~continuous_mask], 2 * binary_history[~continuous_mask] + state_list[~continuous_mask], binary_history[~continuous_mask])
                
                updated_rate[~finished_observations] = np.where(mask&(~np.isnan(rate_if_update_list)), rate_if_update_list, previous_rate_list)
                
                mem_obs[~finished_observations] = observation_index
                mem_bin[~finished_observations] = binary_history
            
            to_return['rate'] = updated_rate
            to_return['obs_idx'] = mem_obs
            to_return['binary_hist'] = mem_bin

            return to_return

    def controlled_force_max_steps(self, fmin:float, fmax:float, Deltat:float)->int:
        self.fmin = fmin
        self.fmax = fmax
        self.Deltat = Deltat
        self.half_interval_force = 4 * (self.Deltat/2)
        return 10000