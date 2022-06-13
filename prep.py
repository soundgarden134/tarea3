import pandas     as pd
import numpy      as np
import math
from timeit import default_timer as timer
from numpy.linalg import eig, eigh, norm, svd
from math import pi


def hankel_features(xe,ye, par_prep):
  M = xe.shape[0]  
  for i in range(0,M):
      print("Muestra numero "+str(i))
      x=xe[i,:]
      get_features(x, par_prep)
   
  return() 

# Hankel's features
def get_features(x, par_prep):
    N = x.shape[0]
    numSegments = par_prep[0]
    L = par_prep[1]
    segments = np.split(x,numSegments)
    J = par_prep[2]
    for idx, segment in enumerate(segments):
        H = get_hankel_matrix(segment, 2,L-1)
        svd_x = hankel_svd(H)
        comps = calculate_features(svd_x, J)
 


    
    return(...)


def get_hankel_matrix(x,rows,cols):
    H = np.zeros((rows,cols))
    i = 0
    row = 0
    for row in range(rows):
        H[row] = x[i:cols+i]
        i += 1
    

    return H

            

def calculate_features(x,J):

    j_comps = np.split(x,J) #supongo que se divide en j componentes despues de esto??


    entropies = np.zeros(J+1)
    for idx,component in enumerate(j_comps):
        entropies[idx] = entropy_spectral(component) #entropia de componentes
    entropies[J] = entropy_spectral(x)
    


    return entropies
        
        
    
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
    start = timer()
    A = np.zeros(len(x))
    
    for k in range(len(x)):
        Ak = 0
        Ak = np.sum(x*np.exp(-2j*pi*(k*np.arange(len(x))/len(x))))

        Ak = norm(Ak)
        A[k] = Ak
        
    end = timer()
    print(end-start)
    return A
    


def p(A, k):
    p_k = A[k]**2/np.sum(A**2)
    return p_k

    
# Binary Label
def binary_label():
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


