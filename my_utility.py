# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
import math  

# Initialize weights
def iniWs(input, nodes,numClasses):   
    W = []
    prev = input
    for n in range(len(nodes)):
        W.append(iniW(nodes[n],prev))
        prev = nodes[n]
    W.append(iniW(numClasses, prev))
    V = initializeV(W)
    return W,V

# Initialize weights for one-layer    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

def initializeV(W): #inicializa matriz de dimension W
    v = [None]*len(W)
    for i in range(len(W)):
        dim1, dim2 = W[i].shape
        v[i] = np.zeros((dim1,dim2))
    return v

# Feed-forward of SNN
def forward(x, w):
    Act=[]
    a0=x.T
    
    z=w[0]@x.T
    a1=bipolar_sigmoid(z)
    
    Act.append(a0)
    Act.append(a1)
    
    ai=a1
    
    for i in range(len(w)):
        if i != 0:
            zi=np.dot(w[i],ai)
            if (i == len(w)-1):
                ai = sigmoid(zi)
            else:
                ai=bipolar_sigmoid(zi)
            Act.append(ai)
    return(Act)     


#Activation function
def sigmoid(x):
    return (1 / (1 + np.exp(-x))).astype(float)

# Derivate of the activation funciton
def deriva_sigmoid(x):
    return sigmoid(x) * (1- sigmoid(x)).astype(float)

def bipolar_sigmoid(x):
    return ((2/(1+np.exp(-x)))-1).astype(float)

def deriva_bipolar(x):
    return (2*np.exp(x))/((np.exp(x)+1)**2).astype(float)
#Feed-Backward of SNN
def gradW(a,w,y):    
    gW = [None]*len(w)
    delta = [None]*len(w)
    
    e = a[-1].T - y
    cost =  ((a[-1].T - y)**2).mean(axis=None) #MSE
    
    delta[-1] = e * deriva_sigmoid(a[-1].T)
    gW[-1] = delta[-1].T @ a[-2].T
    
    for idx in reversed(range(len(w))):
        if idx <= len(w)-2:
            delta[idx] = (w[idx+1].T@delta[idx+1].T*deriva_bipolar(a[idx+1])).T
            gW[idx] = delta[idx].T@a[idx].T
    
    return gW, cost    



# Update Ws RMSProp
def updW(w,v,gW,mu):
    eps = 10**-10
    b = 0.9
    for i in range(len(w)):
        v[i] = b*v[i] + (1-b)*(gW[i])**2
        gRMS_a = (mu/(np.sqrt(v[i] + eps)))
        gRMS = gRMS_a*gW[i]
        w[i] = w[i] - gRMS
    return w,v
    
    

        
        
# Measure
def metricas(x,y):
    
    x = x.T
    confusion_matrix = np.zeros((y.shape[0], x.shape[0]))
    
    for real, predicted in zip(y.T, x.T):
        confusion_matrix[np.argmax(real)][np.argmax(predicted)] += 1
        
    f_score = []
    recall_array = []
    precision_array = []
    accuracy_array = []
    
    TPS = 0
    FPS = 0
    FNS = 0
    TNS = 0
    for index, caracteristica in enumerate(confusion_matrix):
        
        TP = caracteristica[index]
        FP = confusion_matrix.sum(axis=0)[index] - TP
        FN = confusion_matrix.sum(axis=1)[index] - TP
        TN = confusion_matrix.sum(axis=0)[index] - FN
        
        TPS += TP
        FPS += FP
        FNS += FN
        TNS += TN
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        recall_array.append(recall)
        precision_array.append(precision)
        f_score.append(2 * (precision * recall) / (precision + recall))
    precision_array = np.array(precision_array)
    recall_array = np.array(recall_array)
    accuracy = (TPS+TNS)/(TPS+TNS+FPS+FNS)
    print('Accuracy: {:.5f}'.format(accuracy))
    print('Recall promedio: {:.5f}'.format(recall_array.mean()))
    print('Precision promedio: {:.5f}'.format(recall_array.mean()))
    avg_fscore = np.array(f_score).mean()
    f_scores = pd.DataFrame(f_score)
    f_scores = f_scores.append([avg_fscore])
    f_scores.to_csv("fscores.csv", index=False, header=False)
    conf_matrix_df = pd.DataFrame(confusion_matrix)
    conf_matrix_df.to_csv("cmatriz.csv", index=False, header=False)
    f_score = np.array(f_score)

    return(confusion_matrix, f_score) 

    
#Confusion matrix
def confusion_matrix(z,y):
    ...    
    return()
#-----------------------------------------------------------------------
