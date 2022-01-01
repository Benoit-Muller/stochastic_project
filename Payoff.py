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
        """vectorized well if the dimension for w is in the last axe
        Attention,false formula in the pdf: w isn't multiplied by t."""
        return self.S0*np.exp((self.r - self.sigma**2/2)*self.t
                              + self.sigma*w)
    def theta_w(self,w):
        "vectorized"
        return np.sum(self.S(w),axis=-1)/self.m - self.K
    def Psi1(self,w):
        "vectorized well if the dimension for w is in the last axe"
        theta= self.theta_w(w)
        return (theta>0)*theta
    def Psi2(self,w):
        "vectorized well if the dimension for w is in the last axe"
        return 1*(self.Psi1(w)>0)
    def Psi(self,w):
        "vectorized well if the dimension for w is in the last axe"
        Psi1=self.Psi1(w)
        return Psi1, 1*(Psi1>0)
    def normal(self,U):
        """" Transform uniform rvs into iid standard normal rvs;
        Last dimension must be even! 
        Vectorized """
        U[U==0] = 1
        rho = np.sqrt(np.abs(2 * np.log(U[...,:int(self.m/2)])))
        theta = 2 * np.pi * U[...,int(self.m/2):]
        Z=np.zeros(np.shape(U))
        Z[...,:int(self.m/2)]  = rho * np.cos(theta)
        Z[...,int(self.m/2):]  = rho * np.sin(theta)
        return Z
    def normal_bis(self,U): #slower
        return st.norm.ppf(U)
    def weiner(self,Z):
        " Vectorized (the Z-dimension go through the last axe) "
        W= np.zeros(np.shape(Z))
        W[...,0]=self.t[0]*Z[...,0]
        for i in range(1,self.m):
            W[...,i] = W[...,i-1] + (self.t[i] - self.t[i-1])*Z[...,i]
        return W
    def transform1(self,U): #deletable
        " Vectorized "
        return self.Psi1(self.weiner(self.normal(U)))
    def transform(self,U):
        " Vectorized "
        return self.Psi(self.weiner(self.normal(U)))
    def rvs1(self,N): #deletable
        return self.transform1(rnd.uniform(size=(N,self.m)))
    def rvs(self,N):
        return self.transform(rnd.uniform(size=(N,self.m)))
    
    # Define the methods:
    # Question 1:
    def MC1(self,N,alpha=0.01): #deletable
        self.set_data(self.rvs1(N))
        return self.interval(alpha)
    def MC(self,N,alpha=0.1):
        X1,X2 = self.rvs(N)
        return (self.set_data(X1).interval(alpha),
                self.set_data(X2).interval(alpha))
    def QMC1(self,N,K=20,alpha=0.01): #deletable
        X = generate_points(N,self.m)
        U = rnd.uniform(size=(K,1,self.m))
        X= X[None,:,:]
        points = np.floor(X+U)
        Mu=np.mean(self.transform1(points), axis=0)
        return self.set_data(Mu).interval(alpha)
    def QMC(self,N,alpha=0.01,K=20):
        X = generate_points(N,self.m)
        U = rnd.uniform(size=(K,1,self.m))
        X= X[None,:,:]
        Psi1,Psi2 = self.transform((X+U)%1)
        Mu1,Mu2 = np.mean(Psi1, axis=0),np.mean(Psi2, axis=0)
        return (self.set_data(Mu1).interval(alpha), 
                self.set_data(Mu2).interval(alpha))
    # Question 2:
    def theta(self,x,U,position=0):
        U = np.insert(U,position,x,axis=-1)
        return self.theta_w(self.weiner(self.normal(U)))
    def psi(self,U,position=0): #should use ridder or newton method
        fa,fb = ( self.theta(0,U,position=position),
                self.theta(1,U,position=position) )
        if fa>0 and fb>0:
            return 0,1
        if fa<0 and fb<0:
            return 1,1
        c = sp.optimize.root_scalar(self.theta,args=(U,position),maxiter=10,
                                       bracket=[0,1],method="newton").root
        if fa<0:
            return c,1
        else:
            return 0,c
    def integrate(self,U,position=0,order=5):
        integrant=lambda xx: np.array([self.theta(x,U,position)
                                       for x in np.array(xx)])
        a,b = self.psi(U)
        return sp.integrate.fixed_quad(integrant,a,b,n=order)[0],b-a
    def PIMC(self,N,alpha=0.01,order=5,position=0):
        U = rnd.uniform(size=(N,self.m-1))
        Mu = np.array([self.integrate(U[n,:],position,order) for n in range(N)])
        Mu1,Mu2= Mu[:,0],Mu[:,1]
        return (self.set_data(Mu1).interval(alpha), 
                self.set_data(Mu2).interval(alpha))
    def PIQMC(self,N,alpha=0.01,order=5,position=0,K=30):
        X = generate_points(N,self.m-1)
        U = rnd.uniform(size=(K,1,self.m-1))
        X= X[None,:,:]
        XX=(X+U)%1
        Mu =np.array([[self.integrate(XX[k,n,:],position,order) 
                       for n in range(N)] for k in range(K)])
        Mu = np.mean(Mu,axis=1)
        Mu1,Mu2= Mu[:,0],Mu[:,1]
        return (self.set_data(Mu1).interval(alpha), 
                self.set_data(Mu2).interval(alpha))