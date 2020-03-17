# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:52:13 2020

@author: Walter Aburime
"""

import numpy as np

def sigmoid(z):
    
    g = 1/(1 + np.exp(-z))
    
    return g



