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
    """
    Methods for statistics on random variable samples
    Use array of numpy
    """
    def __init__(self,X=np.array([]),ordered=False):
        "Initiate a rv with sample X"
        self.X=X
        self.ordered=ordered
        self.N=len(X)
    def sort(self):
        "Sort the sample to increasing order"
        self.X.sort()
        self.ordered=True
    def cdf(self,x=None):
        """ Empirical cdf of X (evaluated in x)
        if x=None  : return locations of jumps and their heigth
        if x!=None : return the images cdf(x) """
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
        "Alow to add some new data and add it to the sample"
        self.X = np.concatenate((self.X,X))
        self.ordered = False
        self.N = self.N + len(X)
    def set_data(self,X):
        "Alow to set some new data and change the sample"
        self.X = X
        self.ordered = False
        self.N =len(X)
    def mean(self):
        "Compute the empirical esperance"
        return np.sum(self.X)/self.N
    def variance(self,mean=None):
        """Compute the empirical variance,
        using mean if already computed """
        if mean==None:
            mean=self.mean()
        return np.sum((mean - self.X)**2) / (self.N-1)
    def interval(self,alpha):
        """Compute the 1-alpha confidence interval"
        return the mean and the error st. I=[mu +- error] """
        mu= self.mean()
        err =   st.norm.ppf(1-alpha/2) * np.sqrt(self.variance(mu)/self.N)
        return mu, err
            