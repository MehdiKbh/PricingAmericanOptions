import numpy as np
import math
import scipy.stats as st



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
        d1 = (math.log(self.S0/K) + (self.r - self.divs + 0.5*self.sigma*self.sigma)*T)/(self.sigma*math.sqrt(T))
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