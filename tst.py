import pandas as pd
import numpy as np
import my_utility as ut


def save_measure(cm,Fsc):
    
    return()

def load_w_snn(npy_w):
    w = np.load(npy_w, allow_pickle=True)
    return w['arr_0']
  



def load_data_test():
    
    ydata = pd.read_csv("yData.csv",header=None) #solo para contar el numero de clases en Y 
    numClasses = len(ydata.columns)
    
    dtest = pd.read_csv("dtest.csv",header=None)
    dtest = np.array(dtest)
    
    xe = dtest[:,0:-numClasses]
    ye = dtest[:,-numClasses:]
    return xe,ye
    

# Beginning ...
def main():			
    xv,yv  = load_data_test()
    W      = load_w_snn("w_snn.npz")
    zv     = ut.forward(xv,W) 	
    zv = zv[-1] #solo necesitamos el z final
    confusion_matrix,f_score = ut.metricas(yv,zv) 	
    print('F Score promedio: {:.5f}'.format(f_score.mean()))
    print("-------------------------------------")
    print("MATRIZ DE CONFUSION")
    print(confusion_matrix)		
    save_measure(confusion_matrix,f_score)
		

if __name__ == '__main__':   
	 main()

