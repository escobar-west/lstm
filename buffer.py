#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 11:01:26 2017

@author: victor
"""
import numpy as np

class Buffer:
    
    def __init__(self, t_in, t_targ, v_in, v_targ):
        '''
        Args:
            t_in:   N x N_INPUTS training input array
            t_targ: N x N_OUTPUTS
            v_in:   M x N_INPUTS
            v_targ: M x N_OUTPUTS
        '''
        self.t_in = t_in
        self.t_targ = t_targ
        self.v_in = v_in
        self.v_targ = v_targ
    
    def rand_batch(self, batch_size, period):
        '''
        Returns a batch of random inputs and targets.
        Args:
            batch_size: size of batch
            period: period of sample
        Returns:
            inputs: batch_size x period x N_INPUTS array of training inputs
            targets: batch_size x period x N_OUTPUTS array of training targets
        '''
        positions = np.random.randint(0, self.t_in.shape[0] + 1 - period,
                                  size=batch_size)
        
        inputs = [self.t_in[x:x+period,:] for x in positions]
        inputs = np.stack(inputs)
        
        targets = [self.t_targ[x:x+period,:] for x in positions]
        targets = np.stack(targets)
        
        return inputs, targets
            
            