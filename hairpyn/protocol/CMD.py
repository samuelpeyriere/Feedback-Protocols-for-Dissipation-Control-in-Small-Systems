import numpy as np

class CMDProtocol():
    def __init__(self, estimated_number_of_cycles:int=10):
        
        self.name = 'CMD'
        self.f0 = 10 #pN
        self.Deltaf = 1 #pN
        self.rP = self.Deltaf*1e3 #instantaneous jump
        self.rQS = 1 #Quasi static pull rate
        self.estimated_number_of_cycles = estimated_number_of_cycles
        self.initial_rate = 0
    
    def updated_rate(self, time, state_list, previous_rate_list, force_list):
        
        rate_list = np.zeros_like(previous_rate_list)

        maskf0 = (force_list == self.f0)
        maskDeltaf = (force_list == self.f0+self.Deltaf)
        maskrP = (previous_rate_list == self.rP)
        maskrQS = (previous_rate_list == - self.rQS)
        mask0 = (previous_rate_list == 0)
        
        rate_list[(~maskf0)*maskrP] = + self.rP             #continue
        rate_list[(~maskf0)*maskrQS] = - self.rQS           #continue

        rate_list[maskf0*(maskrQS|~mask0)] = 0              #stop pulling
        rate_list[maskf0*state_list*mask0] = + self.rP      #start pulling

        rate_list[maskDeltaf*maskrP] = - self.rQS           #change direction

        return rate_list
    
    def number_of_steps(self, Deltat):
        return self.estimated_number_of_cycles*int(self.Deltaf/Deltat * (1/self.rP + 1/self.rQS)+2)