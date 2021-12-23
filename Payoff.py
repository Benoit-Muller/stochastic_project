#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 10:42:21 2021

@author: benoitmuller
Class for the random variable simulating the payoff
discribed in the application part.
"""

import numpy as np
import scipy as sp
import scipy.stats  as st
import matplotlib.pyplot as plt
import time
import numpy.random as rnd
from mes_stats import RandomVariable
from sobol_new import *

class Payoff(RandomVariable):
    """ Subclass of RandomVariable that simulate the payoff random variable
    discribed in the financial application part.
    """
    def __init__(self,m,K=100,S0=100,r=0.1,sigma=0.1,T=1):
        if m%2==1:
            raise Exception("The dimension m must be even")
        self.m=m
        self.K=K
        self.S0=S0
        self.r=r
        self.sigma=sigma
        self.T=T
        self.t= T/m * np.arange(1,m+1)
        
    # Define functions of the problem, to reduce to a uniform variable:
    def S(self,w):
        return self.S0*np.exp((self.r - self.sigma**2/2)*self.t 
                         + self.sigma*self.t*w)
    def Psi1(self,w):
        return max(np.sum(self.S(w))/self.m - self.K, 0)
    def Psi2(self,w):
        return self.Psi1(w)>0
    def Psi(self,w):
        return self.Psi1(w), self.Psi2(w)
    def normal(self,U):
        """" Transform uniform rvs into normal rvs;
        dimension must be even! """
        U = U + 1*(U==0)
        rho = np.sqrt(-2 * np.log(U[:int(self.m/2)]))
        theta = 2 * np.pi * U[int(self.m/2):]
        Z=np.zeros(np.shape(U))
        Z[:int(self.m/2)]  = rho * np.cos(theta)
        Z[int(self.m/2):]  = rho * np.sin(theta)
        return Z
    def weiner(self,Z):
        W= np.zeros(self.m)
        W[0]=self.t[0]*Z[0]
        for i in range(1,self.m):
            W[i] = W[i-1] + (self.t[i] - self.t[i-1])*Z[i]
        return W
    def transform1(self,U):
        return self.Psi1(self.weiner(self.normal(U)))
    def rvs1(self,N):
        X=np.zeros(N)
        for n in range(N):
            X[n]= self.transform1(rnd.uniform(size=self.m))
        return X
    
    # Define the methods:
    def MC1(self,N,alpha=0.1):
        self.set_data(self.rvs1(N))
        return self.interval(alpha)
    def QMC1(self,N,alpha=0.1):
        U = generate_points(N,self.m,0)
        X=np.zeros(N)
        for n in range(N):
            X[n]= self.transform1(U[n,:])
        self.set_data(X)
        return self.interval(alpha)