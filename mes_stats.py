#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:03:15 2021

@author: benoitmuller
Mes fonctions pour les statistiques
"""
import numpy as np
import scipy.stats  as st
#import matplotlib.pyplot as plt

def cdf(X,x=None):
    """
    Empirical cdf of X evaluated in x
    """
    n=len(X)
    X=np.sort(X)
    if x is None:
        F=np.arange(1,n+1)/n
        X= np.repeat(X, 2)
        F= np.repeat(F, 2)
        F[0:2*n:2]=F[0:2*n:2]-1/n
        return X,F
    else:
        X=np.reshape(X,(n,1))
        return np.sum(X<=x,0)/n
    
class RandomVariable:
    """personal methods for statistics"""
    def __init__(self,X=np.array([]),name="",ordered=False):
        self.X=X
        self.ordered=ordered
        self.N=len(X)
        self.name=name
    def sort(self):
        self.X.sort()
        self.ordered=True
    def cdf(self,x=None):
        """
        Empirical cdf of X (evaluated in x)
        """
        n=len(self.X)
        if self.ordered==False:
            self.sort()
        if x is None:
            F=np.arange(1,n+1)/n
            self.X= np.repeat(self.X, 2)
            F= np.repeat(F, 2)
            F[0:2*n:2]=F[0:2*n:2]-1/n
            return self.X,F
        else:
            self.X=np.reshape(self.X,(n,1))
            return np.sum(self.X<=x,0)/n   
    def add_data(self,X):
        self.X = np.concatenate((self.X,X))
        self.ordered = False
        self.N = self.N + len(X)
    def mean(self):
        return np.sum(self.X)/self.N
    def variance(self):
        return np.sum((self.mean() - self.X)**2) / (self.N-1)
    def interval(self,alpha):
        mu= self.mean()
        delta =   st.norm.ppf(1-alpha/2) * np.sqrt(self.variance()/self.N)
        return mu, delta
            