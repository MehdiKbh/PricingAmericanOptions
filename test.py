# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 01:01:35 2020

@author: Kaabache
"""

import numpy as np
import scipy as sc
import scipy.stats as st
import pandas as pd

import math
import matplotlib.pyplot as plt
import sys

import time

#import StochasticProcess as SP
#import LongstaffSchwarz as LS


#A class for a general Ito Process - Useful for local/stochastic volatility models
class StochasticProcess:
    
    """
        StochasticProcess generates an Ito Process, and enables to compute sample path through a simple Euler recursive scheme
    """
    
    def __init__(self, S0=100.):
        self.S0=S0
        
    def getS0(self):return self.S0
    
    #Function to compute the next step X_t+h given Xt
    def step(self, currentS, h):
        #This function is specific to each model --> to implement in a derived class model(StochasticProcess)
        raise NotImplementedError()
        
    def getTerminalValue(self, T, nbSteps):
        #Compute X_T given X_0, following a simple Euler Scheme
        current_value = self.S0
        h = float(T)/float(nbSteps)
        for k in range(nbSteps):
            current_value = self.step(current_value, h)
        return current_value
    
    def getSamplePath(self, T, nbSteps):
        #Compute all X_t given X_0, following a simple Euler Scheme
        path_values = np.zeros(nbSteps+1)
        path_values[0] = self.S0
        h = float(T)/float(nbSteps)
        for k in range(1,nbSteps+1):
            path_values[k] = self.step(path_values[k-1], h)
        return path_values
        
        
        
#A class specific for the BS model, derived from StochasticProcess
class BS_model(StochasticProcess):
    
    """
        BS_model generates a Black & Scholes Model
        The class is derived of StochasticProcess
    """
    
    def __init__(self, S0=100., vol=.2, r=0., divs=0.):
        self.S0 = S0
        self.sigma = vol
        self.r = r
        self.divs=divs
        
    def getSigma(self):return self.sigma
    def getR(self):return self.r
    def getDivs(self):return self.divs
        
    def step(self, currentS, h):
        #Compute one step ahead (X_t+h given X_t)
        z = math.sqrt(h) * np.random.normal()
        drift = self.r - self.divs - 0.5*self.sigma*self.sigma
        X = currentS * math.exp( drift*h + self.sigma * z )
        return X
    
    #Override the method getTerminalValue with a closed form solution in the case of a geometric BM
    def getTerminalValue(self, T, nbSteps):
        z = math.sqrt(T) * np.random.normal()
        drift = self.r - self.divs - 0.5*self.sigma*self.sigma
        X = self.S0 * math.exp( drift*T + self.sigma * z )
        return X
    
    def computeCallPrice(self, K, T):
        d1 = (math.log(self.S0/K) + (self.r - self.divs - 0.5*self.sigma*self.sigma)*T)/(self.sigma*math.sqrt(T))
        d2 = d1 - self.sigma*math.sqrt(T)
        forward = self.S0 * math.exp((self.r-self.divs)*T)
        discount = math.exp(-self.r*T)
        return ( discount*(forward*st.norm.cdf(d1) - K*st.norm.cdf(d2))  )
    
    def computePutPrice(self, K, T):
        forward = self.S0 * math.exp((self.r-self.divs)*T)
        discount = math.exp(-self.r*T)
        C = self.computeCallPrice(K, T)
        return ( C - discount*(forward - K) )
    
    
    def simulationAnthetic(self, T, nbSteps, nbSimuls):
        #Compute 2*nbSteps paths, using antithetic variables Z and -Z for the generation of the BM
        results = float('inf') * np.ones((2*nbSimuls, nbSteps+1))
        results[:,0] = self.S0
        h = T/float(nbSteps)
        
        for m in range(0, 2*nbSimuls, 2):
            for t in range(1, nbSteps+1):
                z =  np.random.normal()
                drift = self.r - self.divs - 0.5*self.sigma*self.sigma
                results[m,t] = results[m,t-1] * math.exp( drift*h + self.sigma * math.sqrt(h) * z )
                results[m+1,t] = results[m+1,t-1] * math.exp( drift*h + self.sigma * math.sqrt(h) *(-z) )
        return results
    
    
#Class to handle MC on the stochastic process class
#NB - for BS model, an antithetic MC simulation is implemented, which is more efficient
class MC_engine:
    
    def __init__(self, stochastic_model, T=1., nbSteps=100, nbSimuls=100):
        self.model = stochastic_model
        self.T = T
        self.nbSteps = nbSteps
        self.nbSimuls = nbSimuls
        
    def simulationTerminalValues(self):
        results = float('inf') * np.ones(self.nbSimuls)
        for m in range(self.nbSimuls):
            results[m] = self.model.getTerminalValue(self.T, self.nbSteps)
        return results
    
    def averageTerminalValues(self):
        return np.average(self.simulationTerminalValues())
    
    def computeEuropeanPrice(self, payoff, arguments):
        """
            computeEuropeanPrice(self, payoff, arguments) compute the price of a European Contingent Claim
            
                payoff: a Python function that compute the payoff of the ECC at time T.
                        This payoff should be of form payoff(x, arguments), with x the stock_price S_T, and arguments the additional arguments
                    
                arguments: a tuple, that contains all arguments needed in payoff, except x
                
            Example: to compute the price of a call option :
                    X = BS_model()
                    engine = MC_engine(X, 1, 100, 100)
                    def callPrice(x,K): return(max(x-K,0))
                    arg=(100,)
                    engine.computeEuropeanPrice(callPrice, arg)
        """
        stock_prices = self.simulationTerminalValues()
        mean=0.
        for k in range(len(stock_prices)):
            args = (stock_prices[k],) + arguments
            mean += payoff(*args)
        mean *= 1./float(len(stock_prices))
        return mean
    
    def simulation(self):
        results = float('inf') * np.ones((self.nbSimuls, self.nbSteps+1))
        for m in range(self.nbSimuls):
            results[m,:] = self.model.getSamplePath(self.T, self.nbSteps)
        return results
    
    
#Longstaff Schwarz Algo
def LongstaffSchwarz(model, T, nbSteps, nbSimuls, d, basis_projection, payoff, arguments_payoff):
    try:
        X = model.simulationAnthetic(T, nbSteps, nbSimuls//2)
    except:
        engine = MC_engine(model, T, nbSteps, nbSimuls)
        X = engine.simulation()

    T = nbSteps
    M = nbSimuls
    
    r = model.getR()
    #h = float(T)/float(nbSteps)
    
    
    #Initialization
    discount = math.exp(-r*T)
    q = np.zeros(M)  #Continuation value ( E[U(X_t+1) / X_t] )
    Z = discount * numpyPayoff(payoff, X[:,T], arguments_payoff)  #Payoff at time t
    U = discount * numpyPayoff(payoff, X[:,T], arguments_payoff)  #Price at time t for each path
    
   
    for t in range(T-1, 0, -1):
        #Discount the value for each path
        dscount = math.exp(-r*t)
        U = discount * U
        
        #Compute the least square coefficients and the continuation value
        alphas = get_alphas(X[:,t], U, d, basis=basis_projection)[0]
        proj = projection(d, basis=basis_projection, coefficients=alphas)
        q = proj.getProjection(X[:,t]) #New continuation value
        
        #Update the payoff at time t
        Z = discount * numpyPayoff(payoff, X[:,t], arguments_payoff)
       
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

def basis_poly(x,k):
    return np.power(x,k)

def callPrice(x,k): return(max(x-k,0))

def Laguerre_polyn(x, k):
    if(k<0):
        sys.exit("k should be a positive integer")
    if(k==0):
        return 1
    elif (k==1):
        return 1-x
    else:
        n = k-1
        return ( -((x-2*n - 1)*Laguerre_polyn(x,k-1) + n*Laguerre_polyn(x,k-2))/float(n+1) )
    
def Laguerre_polyn_exp(x,k):
    return math.exp(-x/2)*Laguerre_polyn(x,k)

start = time.time()
print(LongstaffSchwarz(BS_model(), T=1, nbSteps=100, nbSimuls=10000, d=3, basis_projection=Laguerre_polyn, payoff=callPrice, arguments_payoff=(100,)))
print("Running Time: ", time.time()-start)

start = time.time()
print(LongstaffSchwarz(BS_model(), T=1, nbSteps=100, nbSimuls=10000, d=3, basis_projection=basis_poly, payoff=callPrice, arguments_payoff=(100,)))
print("Running Time: ", time.time()-start)