import numpy as np
from scipy.optimize import root_scalar

class DNAHandles():
    def __init__(self,
                 handle_persistence_length:float=10.0,
                 handle_contour_length:float=19.72)->None:
        """handle_persistence_length and handle_contour_length are all in nm"""

        self.beta = 0.24 #(pN.nm)^-1

        self.xmin_int = 1e-5

        self.handle_persistence_length = handle_persistence_length
        self.handle_contour_length = handle_contour_length

    def force(self, state:np.ndarray, length:np.ndarray)->np.ndarray:
        """
        Takes the total length(s) of both DNA handles, returns the force(s) stretching them according to the WLC model.
    
        Parameters:
        - state (ndarray): The state(s) of the DNA hairpin(s) (Folded:False or Unfolded:True)
        - length (ndarray): The total length(s) of both DNA handles (in nm)
    
        Returns:
        ndarray: The force(s) stretching both DNA handles (in pN)
        """
        return self.beta/(4*self.handle_persistence_length) * ((1-length/self.handle_contour_length)**-2 + 4 * length/self.handle_contour_length -1)

    def length_equation(self, length:np.ndarray, state:np.ndarray, force:np.ndarray)->np.ndarray:
        return self.force(state, length) - force
    
    def length(self, state:np.ndarray, force:np.ndarray)->np.ndarray:
        """
        Takes the force(s) stretching both DNA handles, returns their total length according to the WLC model.
    
        Parameters:
        - state (ndarray): The state(s) of the DNA hairpin(s) (Folded:False or Unfolded:True)
        - force (ndarray): The force(s) stretching both DNA handles (in pN)
    
        Returns:
        ndarray: The total length(s) of both DNA handles (in nm)
        """

        length = np.zeros_like(force)

        for (i, force_element), (_, state_element) in zip(np.ndenumerate(force), np.ndenumerate(state)):
            length[i] = root_scalar(self.length_equation, bracket=[self.xmin_int, self.handle_contour_length-1e-5], args=(state_element, force_element,)).root
        return length