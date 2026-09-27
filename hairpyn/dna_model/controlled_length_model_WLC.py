import mpmath as mp
import numpy as np
from hairpyn.dna_model.dna_hairpin import DNAHairpin
from hairpyn.dna_model.dna_handles import DNAHandles
from hairpyn.dna_model.trapped_bead import TrappedBead
from scipy.optimize import root_scalar
from scipy.integrate import quad

class ControlledLengthModelWLC():
    def __init__(self):

        self.beta = 0.24 #(pN.nm)^-1

        self.DeltaG0 = 264.0 #pN.nm
        self.k0 = 2e-14 #1/s
        self.xm = 18 #nm

        self.DeltaG0t = 0 #pN.nm  #barrier height without force (it is included in k0)
        self.xt = 9 #nm

        self.DeltaG0star = self.DeltaG0t - self.DeltaG0
        self.xstar = self.xm - self.xt

        self.fc = self.DeltaG0/self.xm #à revoir
        self.kc = self.kFtoU(self.fc)

        self.DNA_hairpin = DNAHairpin()
        self.DNA_handles = DNAHandles()
        self.trapped_bead = TrappedBead()

        self.fmin_int = 1e-5 #pN
        self.fmax_int = 40 #pN
        
        self.xmax = 600 #nm
        self.xmin = 1e-2 #nm

        self.xmin_int = 1e-2 #nm
    
    def DeltaGFU(self, force:np.ndarray)->np.ndarray:
        xU_DNA = self.DNA_hairpin.length(state=np.full_like(force, True, dtype='bool'), force=force)
        xF_DNA = self.DNA_hairpin.length(state=np.full_like(force, False, dtype='bool'), force=force)
        DeltaW_DNA = np.array([quad(lambda x: self.DNA_hairpin.force(state=np.array([True]), length=np.array([x]))[0], xU_DNA[k], self.xmin_int)[0] - quad(lambda x: self.DNA_hairpin.force(state=np.array([False]), length=np.array([x]))[0], xF_DNA[k], self.xmin_int)[0] for k in range(len(force))])

        xU_h = self.DNA_handles.length(state=np.full_like(force, True, dtype='bool'), force=force)
        xF_h = self.DNA_handles.length(state=np.full_like(force, False, dtype='bool'), force=force)
        DeltaW_h = np.array([quad(lambda x: self.DNA_handles.force(state=np.array([None]), length=np.array([x]))[0], xU_h[k], xF_h[k])[0] for k in range(len(force))])

        xU_b = self.trapped_bead.displacement(state=np.full_like(force, True, dtype='bool'), force=force)
        xF_b = self.trapped_bead.displacement(state=np.full_like(force, False, dtype='bool'), force=force)
        DeltaW_b = np.array([quad(lambda x: self.trapped_bead.force(state=np.array([None]), length=np.array([x]))[0], xU_b[k], xF_b[k])[0] for k in range(len(force))])

        return self.DeltaG0 + DeltaW_DNA + DeltaW_h + DeltaW_b

    def length(self, state:np.ndarray, force:np.ndarray)->np.ndarray:
        # if np.min(force) < 1e-7:
        #     raise Exception('The constraints for inversion of the equations are such that force > 1e-7 pN.')
        return self.DNA_hairpin.length(state, force) + self.DNA_handles.length(state, force) + self.trapped_bead.displacement(state, force)
    
    def equationS1(self, force:np.ndarray, state:np.ndarray, length:np.ndarray)->np.ndarray:
        return self.length(state, force) - length
    
    def force(self, state:np.ndarray, length:np.ndarray)->np.ndarray:
        force = np.zeros_like(length)
        for (i, length_element), (_, state_element) in zip(np.ndenumerate(length), np.ndenumerate(state)):
            # if length[i] < 1e-2 or length[i] > 600:
            #     raise Exception('The constraints for inversion of the equations limit the total length to more than 0.01 nm and less than 600 nm.')
            force[i] = root_scalar(self.equationS1, bracket=[self.fmin_int, self.fmax_int], args=(state_element, length_element,)).root
        return force
    
    def B(self, force:np.ndarray)->np.ndarray:
        return -force*self.xt
    
    def kFtoU(self, f:np.ndarray)->np.ndarray:
        """
        Takes the force(s) stretching the DNA hairpin(s), returns the probability rate(s) to hop from F to U.
    
        Parameters:
        - force (ndarray): The force(s) stretching the DNA hairpin(s) (in pN)
    
        Returns:
        float: the probability rate(s) to hop from F to U
        """
        return self.k0 * np.exp(- self.beta * self.B(f))
    
    def kUtoF(self, force:np.ndarray)->np.ndarray:
        """
        Takes the force(s) stretching the DNA hairpin(s), returns the probability rate(s) to hop from U to F.
    
        Parameters:
        - force (float or ndarray): The force(s) stretching the DNA hairpin(s) (in pN)
    
        Returns:
        float: the probability rate(s) to hop from U to F
        """
        return self.k0 * np.exp(self.beta * (self.DeltaGFU(force) + self.B(force)))