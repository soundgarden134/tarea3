# SNN's Training :

import pandas     as pd
import numpy      as np
import my_utility as ut


#Save weights of the SNN
def save_w():
    ...
    return
    
#SNN's Training 
def train(x,y,param):    
    W,V,S = iniWs()
    Costo = []
    for iter in range(MaxIter):        
        a       = ut.forward(...)
        gW,cost = ut.gradW(...)        
        W,V,S   = ut.updW(...)
        Costo.append(Cost)
    return(W,Costo)

# Load data from xData.csv, yData,csv
def load_data_trn():
    ....
    return()
    
# Load parameters for SNN'straining
def load_cnf_snn():
    ...
    return()
   
# Beginning ...
def main():
    param       = load_cnf_snn()            
    xe,ye       = load_data_trn()   
    W,Cost      = snn_train(xe,ye,param)             
    save_w(W,Cost)
       
if __name__ == '__main__':   
	 main()

