import pandas     as pd
import numpy      as np
import math
from timeit import default_timer as timer
from numpy.linalg import eig, eigh, norm, svd
from math import pi


def hankel_features(xe,ye, par_prep):
  M = xe.shape[0]  
  numSegments = par_prep[0]
  new_x = []
  new_y = []
  start = timer()
  for i in range(0,M):
      print("Muestra numero "+str(i+1))
      x=xe[i,:]
      new_samples,new_labels = get_features(x, par_prep, ye[i])
      if i == 0:
          new_x = new_samples
          new_y = new_labels
      else:
          new_x = np.vstack([new_x, new_samples])
          new_y = np.vstack([new_y, new_labels])
  end = timer()
  print("Elapsed time: " + str(round(end-start)) + " seconds.")
  return new_x, new_y 

# Hankel's features
def get_features(x, par_prep, label):
    N = x.shape[0]
    numSegments = par_prep[0]
    L = par_prep[1]
    segments = np.split(x,numSegments)
    J = par_prep[2]
    new_features = []
    new_labels = []
    for idx, segment in enumerate(segments):
        j_comps = np.split(segment, J)
        new_x = []
        comp_matrix = []
        for comp in j_comps:
            length = len(comp)
            H = get_hankel_matrix(comp, 2,length-1)
            svd_x = hankel_svd(H)
            comp_matrix = np.append(comp_matrix,svd_x)
            entropy = entropy_spectral(svd_x)
            new_x.append(entropy)
        comp_matrix = np.reshape(comp_matrix, (len(comp)*2,J))
        _,S,_ = svd(comp_matrix) #valores singulares
        seg_entropy = entropy_spectral(segment)
        new_x.append(seg_entropy)
        new_x = np.concatenate((new_x,S))
        new_features = np.append(new_features, new_x)


    end = timer()
    new_features = np.split(new_features, numSegments)
    for i in range(len(segments)):
        new_labels = np.append(new_labels,label)
    new_labels = np.reshape(new_labels, (len(segments),len(label)))
        
 


    
    return new_features, new_labels


def get_hankel_matrix(x,rows,cols):
    H = np.zeros((rows,cols))
    i = 0
    row = 0
    for row in range(rows):
        H[row] = x[i:cols+i]
        i += 1
    

    return H

            

        
    
def hankel_svd(H): #retorna el x de la suma de c1 y c2
    U, S, V = svd(H, full_matrices = False)
    V = V.T
    
    a = U[:,0].reshape(U.shape[0],1) 
    b = V[:,0].reshape(1,V.shape[0])
    h1 = S[0]*a@b
    a = U[:,1].reshape(U.shape[0],1)
    b = V[:,1].reshape(1,V.shape[0])
    h2 = S[1]*a@b
    #FEO PERO FUNCIONA

    c1 = h1[0,:]
    c1 = np.append(c1,h1[1,-1])

    c2 = h2[0,:]
    c2 = np.append(c2,h2[1,-1])
    
    x = np.append(c1,c2)
    
    return x





# spectral entropy
def entropy_spectral(X):

    H = 0
    A = get_a(X)

    for k in range(len(X)):
        P = p(A,k)
        H += P*np.log2(P)
        
    H = (-1/np.log2(len(X))) * H

    return H

def get_a(x):
    A = np.zeros(len(x))
    
    for k in range(len(x)):
        Ak = 0
        Ak = np.sum(x*np.exp(-2j*pi*(k*np.arange(len(x))/len(x))))

        Ak = norm(Ak)
        A[k] = Ak
        

    return A
    


def p(A, k):
    p_k = A[k]**2/np.sum(A**2)
    return p_k

    
# Binary Label
def binary_label(): #uso get dummies
  ...
  return

# Data norm 
def data_norm():
  ...
  return 

# Save Data from  Hankel's features
def save_data_features(Dinp, Dout):
    ...  
    return

# Load data from Data.csv
def load_data(fname):
    data = pd.read_csv(fname,header=None)
    xe = data.iloc[:,:-1]
    xe = np.array(xe)
    ye = data.iloc[:, -1]
    ye = pd.get_dummies(ye) #one hot encoder
    ye = np.array(ye)


    return(xe,ye)

# Parameters for pre-proc.
def load_cnf_prep():
    params = np.genfromtxt("cnf_prep.csv",delimiter=',')    
    par_prep=[]    
    par_prep.append(np.int16(params[0])) # numSegmentos
    par_prep.append(np.int16(params[1])) # LongitudSegmento
    par_prep.append(np.int16(params[2])) # NumComponentes

    return par_prep

# Beginning ...
def main():        
    par_prep    = load_cnf_prep()	
    print("Cargando datos")
    xe, ye      = load_data("Data_1.csv")	
    print("Datos cargados")
    Dinput,Dout = hankel_features(xe,ye, par_prep)
    # Dinput      = data_norm(Dinput)
    # save_data_features(Dinput,Dout)


if __name__ == '__main__':   
	 main()


