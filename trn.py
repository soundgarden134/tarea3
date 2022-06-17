# SNN's Training :

import pandas     as pd
import numpy      as np
import my_utility as ut


#Save weights of the SNN
def save_w():
    ...
    return


    
#SNN's Training 
def snn_train(x,y,param):    
    numClasses = y.shape[1]
    trainingPerc = param[0]
    MaxIter = param[1]
    learningRate = param[2]
    W,V = ut.iniWs(x.shape[1], param[3:],numClasses) #nodos de entrada y nodos ocultos, falta S
    Costo = []
    for iter in range(MaxIter):        
        a       = ut.forward(x,W)
        gW,cost = ut.gradW(a,W,y)        
        W,V   = ut.updW(W,V,gW,learningRate)
        Costo.append(cost)
    return(W,Costo)

# Load data from xData.csv, yData,csv
def load_data_trn(trainingPerc):
    
    xdata = pd.read_csv("xData.csv",header=None)
    xe = np.array(xdata)
    
    ydata = pd.read_csv("yData.csv",header=None)
    ye = np.array(ydata)
    
    numClasses = ye.shape[1]
    
    data = np.append(xe,ye, axis = 1)
    
    np.random.shuffle(data) #reordena la data aleatoriamente
    
    data_division = int(len(data)*trainingPerc) #para dejar 70% train 30% test
    
    dtrain = data[0:data_division]
    dtest = data[data_division:]
    
    np.savetxt("dtrain.csv", dtrain, delimiter = ',')
    np.savetxt("dtest.csv", dtest, delimiter = ',')
    
    xe = dtrain[:,0:-numClasses]
    ye = dtrain[:,-numClasses:]
    
    return(xe,ye)

    
# Load parameters for SNN'straining
def load_cnf_snn():
    par = np.genfromtxt("cnf_snn.csv",delimiter=',')    
    par_snn = []   
    par_snn.append(float(par[0])) # trainingPerc
    par_snn.append(np.int16(par[1])) # maxIter
    par_snn.append(float(par[2])) # LearningRate
    for i in range(3,len(par)):
        par_snn.append(np.int16(par[i])) #nodos

    return par_snn

   
# Beginning ...
def main():
    param       = load_cnf_snn()            
    xe,ye       = load_data_trn(param[0])   
    W,Cost      = snn_train(xe,ye,param)    
    print(Cost[0])
    print(Cost[200])
    print(Cost[-1])      

    # save_w(W,Cost)
       
if __name__ == '__main__':   
	 main()

