import numpy as np
import scipy as sc
import scipy.stats as st

import math
import sys

import StochasticProcess

#Longstaff Schwarz Algo
def LongstaffSchwarz(model, T, nbSteps, nbSimuls, d, basis_projection, payoff, arguments_payoff):
    try:
        X = model.simulationAnthetic(T, nbSteps, nbSimuls//2)
    except:
        engine = MC_engine(model, T, nbSteps, nbSimuls)
        X = engine.simulation()
    
    r = model.getR()
    h = float(T)/float(nbSteps)
    
    
    #Initialization
    #discount = math.exp(-r*T)
    q = np.zeros(nbSimuls)  #Continuation value ( E[U(X_t+1) / X_t] )
    Z = numpyPayoff(payoff, X[:,nbSteps], arguments_payoff)  #Payoff at time t
    U = numpyPayoff(payoff, X[:,nbSteps], arguments_payoff)  #Price at time t for each path
    
   
    for t in range(nbSteps-1, 0, -1):
        #Discount the value for each path
        discount = math.exp(-r*h)
        U = discount * U
        
        #Compute the least square coefficients and the continuation value
        alphas = get_alphas(X[:,t], U, d, basis=basis_projection)[0]
        proj = projection(d, basis=basis_projection, coefficients=alphas)
        q = proj.getProjection(X[:,t]) #New continuation value
        
        #Update the payoff at time t
        Z = numpyPayoff(payoff, X[:,t], arguments_payoff)
       
        #Exercice on the set (Z>=q)
        U = numpyIndicSup(Z,q)*Z + (1-numpyIndicSup(Z,q))*U
    
    return np.average(U)
    


#A class to handle projection on a basis of L2
class projection:
    
    def __init__(self, d, basis, coefficients, arguments=()):
        
        """
        projection represents and compute the projection on a certain orthogonal basis of L2 (truncated at the order d)
        
        Arguments:
            d: (type int) Order of truncature
            basis: (type function) A function that gives the basis as a result
            coefficients: (type ndarray) The coefficients of the projection with respect to each vector of the basis
            arguments: (type tuple) Additionnal arguments needed to compute the vectors of the basis in the function basis 
            
        """
        
        self.dim = d
        self.coeffs = coefficients
        self.basis = basis
        self.args = arguments
        
    def getCoefficients(self):
        """
            getCoefficients() return the list of coefficients of the projection
        
        """
        return self.coeffs 
    
    def getBasisValue(self, x, k):
        """
            getBasisValue(x,k) return the value of the basis vector k evaluated at x
            
            Input:
                x: (type int or float or ndarray) the point at which we evaluate the basis vector function k
                k: the number of the basis vector function that we want to evaluate. k should be between 0 and d-1
                
            Output:
                int or float or ndarray reprezenting the evaluation of the vector basis function k at x
        """
        
        #Check if the basis vector is defined in the truncated basis
        if(k>=self.dim or k<0): 
            sys.exit("The basis function number "+str(k)+" is not defined -- You should choose a basis function between 0 and d-1")
            
        else:
            #Check if x is an array
            if(type(x) == np.ndarray):
                res = np.zeros(len(x))
                for i in range(len(x)):
                    #Compute the projection for each entry of x
                    res[i] = self.basis( *( (x[i],k,) +self.args ) )
                return res
            else:
                #If x is not an array, we can directly compute the projection
                return self.basis( *( (x,k,)+self.args ) )
            
            
    def getVectorBasis(self, x):
        """
            getVectorBasis(x) return the value of the all the basis vectors evaluated at x
            
            Input:
                x: (float or int or ndarray) the point at each we want to evaluate all the basis vectors
                
            Output:
                Ndarray of all the vector basis functions evaluated at x.
                If x is an array, each row k (k=0,...,d-1) of the output is the evaluation of x by the basis vector function k
        """
        return ( np.array([self.getBasisValue(x, k) for k in range(self.dim)]) )
    
    def getProjection(self, x):
        """
            getProjection(x) compute the projection of x into the basis defined by basis, with respect to the coefficients coefficients
            
            Input:
                x: (float or int or ndarray) the point at each we want to evaluate the projection
                
            Output:
                Float or ndarray. Projection of all elements of X.
        """
        X = self.getVectorBasis(x)
        return (np.dot(self.coeffs, X))
        
        
        
#Function to find the least square estimator        
def get_alphas(X,Z,d, basis):
    if(len(X)!=len(Z)): sys.exit("X and Z should be of same size")
    else:
        proj = projection(d, basis=basis, coefficients=np.ones(d))
        N = len(X)
        #Creation of the matrix A -- To adapt to make it vectorized
        A = float('inf')*np.ones((d,d))
        for k in range(d):
            for l in range(d):
                A[k,l] = np.average(proj.getBasisValue(X,k)*proj.getBasisValue(X,l))
                
                
        #Creation of Psi -- To adapt to make it vectorized
        psi = np.zeros(d)
        for n in range(N):
            psi = psi + Z[n] * proj.getVectorBasis(X[n])
        psi *= 1./float(N)
        
        #Computation of alpha
        alpha = np.linalg.lstsq(A, psi, rcond=None)
        
        return alpha
        
        
#Some useful functions for the LS Algo - Not implemented in numpy i think
def numpyMax(a,b):
    
    """
       numpyMax(a,b) returns the max of a and b entry by entry
       
       Inputs: Two numpy.ndarray a and b of same size
       Outputs: A numpy.ndaray (shape = a.shape) that entries are the maximum of the corresponding entries in a and b 
    """
    
    if not(type(a).__module__ == np.__name__ and type(a).__module__ == np.__name__):
        sys.exit("a and b should be numpy objects")
    if(len(a)!=len(b)): sys.exit("a and b should be of same size")
    else:
        res = np.copy(a)
        for i in range(len(a)):
            res[i] += (a[i]<b[i])*(b[i] - a[i]) 
        return res

    

def numpyPayoff(payoff, X, arguments):
    
    """
       numpyPayoff(payoff, X, arguments) returns an array of same size than X, which entries are the payoff computed at each
       corresponding entry in X
       
       Inputs: 
           payoff : (type function) payoff(x, arguments) - a function that computes a certain payoff (returns a float) at the point x
           X: (type array) 1d-array of each points at which we want to compute the payoff 
           arguments: (type tuple) additionnal arguments needed to compute the payoff
           
       Outputs: 1d-array of size len(X), which entries are the payoff computed at each entry of X 
    """
    Z=float('inf')*np.ones(len(X))
    for k in range(len(X)):
        args = (X[k],)+arguments
        Z[k] = payoff(*args)
    return Z



def numpyIndicSup(a,b):
    
    
    """
       numpyIndicSup(a,b) returns, for each corresponding entry of a and b, 1 if a[i]>b[i], 0 otherwise 
       
       Inputs: Two numpy.ndarray a and b of same size
       Outputs: A numpy.ndaray (shape = a.shape) that entries are the indicator function of the corresponding entries in a and b 
    """
    
    res = np.zeros(len(a))
    for i in range(len(a)):
        res[i] = (a[i]>=b[i])
    return res