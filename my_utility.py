# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
  

# Initialize weights
def iniWs():    
    ...
    return()

# Initialize weights for one-layer    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Feed-forward of SNN
def forward():
    ...    
    return() 

#Activation function
def act_function():
    ...
    return()   
# Derivate of the activation funciton
def deriva_act():
    ...
    return(...)

#Feed-Backward of SNN
def gradW(...):    
    ...    
    return()    

# Update Ws
def updW(...):
    ...    
    return(...)

# Measure
def metricas(x,y):
    ...    
    return()
    
#Confusion matrix
def confusion_matrix(z,y):
    ...    
    return(cm)
#-----------------------------------------------------------------------
