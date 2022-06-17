import pandas as pd
import numpy as np
import my_utility as ut


def save_measure(cm,Fsc):
    ...
    return()

def load_w_snn(npy_w):
    w = np.load(npy_w, allow_pickle=True)
    return w['arr_0']
  



def load_data_test():
    
    dtest = pd.read_csv("dtest.csv",header=None)
    dtest = np.array(dtest)
    
    xe = dtest[:,0:-4]
    ye = dtest[:,-4:]
    return xe,ye
    

# Beginning ...
def main():			
    xv,yv  = load_data_test()
    W      = load_w_snn("w_snn.npz")
    zv     = ut.forward(xv,W) 	
    zv = zv[-1] #solo necesitamos el z final
    confusion_matrix,f_score = ut.metricas(yv,zv) 		
# 	cm,Fsc = ut.metricas(yv,zv) 	
# 	save_measure(cm,Fsc)
		

if __name__ == '__main__':   
	 main()

