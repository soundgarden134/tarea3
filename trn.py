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
    data = pd.read_csv("dtrain.csv",header=None)
    xe = data.iloc[:-1, :]
    xe = np.array(xe)
    ye = data.iloc[-1, :]
    ye = pd.get_dummies(ye) #one hot encoder
    ye = np.array(ye)
    ye = ye.T

    return(xe,ye)
    return()
    
# Load parameters for SNN'straining
def load_cnf_snn():
    par = np.genfromtxt("param_snn.csv",delimiter=',')    
    par_snn = []   
    par_snn.append(np.int16(par[0])) # trainingPerc
    par_snn.append(np.int16(par[1])) # maxIter
    par_snn.append(np.float(par[2])) # LearningRate
    for i in range(3,len(par)):
        par_snn.append(np.int16(par[i])) #nodos

    return par_snn

   
# Beginning ...
def main():
    param       = load_cnf_snn()            
    xe,ye       = load_data_trn()   
    W,Cost      = snn_train(xe,ye,param)             
    save_w(W,Cost)
       
if __name__ == '__main__':   
	 main()

