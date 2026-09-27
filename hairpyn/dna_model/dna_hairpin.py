import numpy as np
from scipy.optimize import root_scalar

class DNAHairpin():
    def __init__(self,
                 unfolded_persistence_length:float=1.34,
                 unfolded_contour_length:float=25.96, folded_dipole_length:float=2.0)->None:
        """unfolded_persistence_length, unfolded_contour_length and folded_dipole_length are all in nm"""
        
        self.fmax_int = 40 # just made a plot to check
        self.fmin_int = 1e-5 #idem
        self.xmin_int = 1e-5

        self.beta = 0.24 #(pN.nm)^-1

        self._cached_results = {}  # Cache for memoization

        self.unfolded_persistence_length = unfolded_persistence_length
        self.unfolded_contour_length = unfolded_contour_length
        self.folded_dipole_length = folded_dipole_length
    
    def force_equation(self, force:np.ndarray, length:np.ndarray)->np.ndarray:
        return self.length(np.full_like(force, False, dtype=bool), force) - length
    
    def force(self, state:np.ndarray, length:np.ndarray)->np.ndarray:
        """
        Takes the length(s) of the DNA hairpin(s), returns the force(s) stretching it(them) according to the dipole and WLC model.
        
        Parameters:
        - state (ndarray): The state(s) of the DNA hairpin (Folded:False or Unfolded:True)
        - length (ndarray): The length(s) of the DNA hairpin (in nm)
        
        Returns:
        - ndarray: The force(s) stretching the DNA hairpin (in pN)
        """

        forces = np.zeros_like(length)

        # Unfolded
        if np.isscalar(length):
            if state:
                return np.array(self.beta/(4*self.unfolded_persistence_length) * ((1-length/self.unfolded_contour_length)**-2 + 4 * length/self.unfolded_contour_length-1))
        else:
            forces[state] = self.beta/(4*self.unfolded_persistence_length) * ((1-length[state]/self.unfolded_contour_length)**-2 + 4 * length[state]/self.unfolded_contour_length-1)

        # Folded
        for (i, length_element), (_, state_element) in zip(np.ndenumerate(length), np.ndenumerate(state)):
            if state_element == False:
                #if length_element >= self.folded_dipole_length or length_element <= 0:
                #    raise Exception('The DNA hairpin folded length must be superior to 0 nm and inferior to ' + str(np.round(self.folded_dipole_length,2)) + ' nm.') #in reality 1.9 (condition force <40pN)
                forces[i] = root_scalar(self.force_equation, bracket=[self.fmin_int, self.fmax_int], args=(length_element,)).root
        return forces
    
    def length_equation(self, length:np.ndarray, force:np.ndarray)->np.ndarray:
        return self.force(np.full_like(length, True, dtype=bool), length) - force

    def length(self, state:np.ndarray, force:np.ndarray)->np.ndarray:
        """
        Takes the force(s) stretching the DNA hairpin(s), returns its(their) length(s) according to the dipole and WLC model.
    
        Parameters:
        - state (ndarray): The state(s) of the DNA hairpin(s) (Folded:False or Unfolded:True)
        - force (ndarray): The force(s) stretching the DNA hairpin(s) (in pN)
    
        Returns:
        ndarray: The length(s) of the DNA hairpin(s) (in nm)
        """

        lengths = np.zeros_like(force)
        
        #Folded
        if np.isscalar(force):
            if state == False:
                return np.array(self.folded_dipole_length*(1/np.tanh(self.folded_dipole_length*self.beta*force)-1/(self.folded_dipole_length*self.beta*force)))
        else:
            lengths[~state] = self.folded_dipole_length*(1/np.tanh(self.folded_dipole_length*self.beta*force[~state])-1/(self.folded_dipole_length*self.beta*force[~state]))
            
        #Unfolded
        for (i, force_element), (_, state_element) in zip(np.ndenumerate(force), np.ndenumerate(state)):
            if state_element: #Unfolded
                lengths[i] = root_scalar(self.length_equation, bracket=[self.xmin_int, self.unfolded_contour_length-1e-5], args=(force_element,)).root
        return lengths