import numpy as np
import numpy.linalg as la

#This function calculates the SVD and returns S' a resized version of S matrix
def SVD(matrix, dimension):
    A = matrix
    r = dimension
    
    AT = A.T
    AAT = np.dot(A, AT)
    evalues1, evectors1 = la.eig(AAT)
    
    #calculate out the S matrix
    nonzero = evalues1[evalues1 !=0]
    shape = np.shape(A)
    D = np.diag(nonzero)
    D.resize(shape) 
    
    #calculate U and V
    ATA = np.dot(AT,A) 
    evalues2, evectors2 = la.eig(ATA)
    
    #U=evectors1, V=evectors2, S=np.sqrt(D)
    #resize U, V, & S to the r value, to get U', S', & V'
    U1 = evectors1[0:(np.size(evectors1, 0)), 0:r]
    V1 = evectors2[0:(np.size(evectors2,0)), 0:r]
    S1 = np.sqrt(D)[0:r, 0:r]
    
    print("U': \n", U1, "\n") 
    print("S': \n",S1, "\n") 
    print("V': \n", V1)
    
    return S1 

A = np.array([[2,4],[1,3],[0,0],[0,0]])
r = 2
S2 = SVD(A, r)
print("S' \n",S2)