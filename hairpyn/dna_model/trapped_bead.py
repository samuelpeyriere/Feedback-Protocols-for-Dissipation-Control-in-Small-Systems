import numpy as np

class TrappedBead():
    def __init__(self,
                 kb:float=0.068)->None:
        """kb is in pN/nm"""

        self.kb = kb
    
    def force(self, state:np.ndarray, length:np.ndarray)->np.ndarray:
        """
        Takes the diplacement(s) of the bead, returns the force(s) pulling on it according to Hooke's law.
    
        Parameters:
        - state (ndarray): The state(s) of the DNA hairpin(s) (Folded:False or Unfolded:True)
        - displacement (ndarray): The bead's displacemement(s) (in nm)
    
        Returns:
        ndarray: The force(s) pulling on the bead (in pN)
        """
        return self.kb * length
    
    def displacement(self, state:np.ndarray, force:np.ndarray)->np.ndarray:
        """
        Takes the force(s) pulling on the bead, returns its(their) displacement(s) according to Hooke's law.
    
        Parameters:
        - state (ndarray): The state(s) of the DNA hairpin(s) (Folded:False or Unfolded:True)
        - force (ndarray): The force(s) pulling on the bead (in pN)
    
        Returns:
        ndarray: The bead's displacemement(s) (in nm)
        """
        return force / self.kb