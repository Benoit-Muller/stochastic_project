#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:03:15 2021

@author: benoitmuller
Mes fonctions pour les statistiques
"""
import numpy as np
import scipy.stats as st
from sobol_new import *
import numpy.random as rnd

# import matplotlib.pyplot as plt


def cdf(X, x=None):
    """
    Empirical cdf of X evaluated in x
    """
    n = len(X)
    X = np.sort(X)
    if x is None:
        F = np.arange(1, n + 1) / n
        X = np.repeat(X, 2)
        F = np.repeat(F, 2)
        F[0 : 2 * n : 2] = F[0 : 2 * n : 2] - 1 / n
        return X, F
    else:
        X = np.reshape(X, (n, 1))
        return np.sum(X <= x, 0) / n


class RandomVariable:
    """
    Methods for statistics on scalar or vector(iid components) random variable samples
    Use arrays of numpy
    """

    def __init__(self, X=np.array([]), ordered=False):
        "Initiate a rv with sample X"
        self.X = X
        self.ordered = ordered

    def sort(self):
        "If scalar, sort the sample to increasing order"
        if self.ordered == False:
            self.X.sort()
            self.ordered = True

    def cdf(self, x=None):
        """If scalar, Empirical cdf of X (evaluated in x)
        if x=None  : return locations of jumps and their heigth
        if x!=None : return the images cdf(x) """
        n = len(self.X)
        self.sort()
        if x is None:
            F = np.arange(1, n + 1) / n
            self.X = np.repeat(self.X, 2)
            F = np.repeat(F, 2)
            F[0 : 2 * n : 2] = F[0 : 2 * n : 2] - 1 / n
            return self.X, F
        else:
            self.X = np.reshape(self.X, (n, 1))
            return np.sum(self.X <= x, 0) / n

    def add_data(self, X):
        "Allow to add some new data and add it to the sample"
        self.X = np.concatenate((self.X, X), axis=0)
        self.ordered = False
        self.N = np.shape(self.X)[0]
        return self

    def set_data(self, X):
        "Alow to set some new data and change the sample"
        self.X = X
        self.ordered = False
        self.N = np.shape(X)[0]
        return self

    def mean(self):
        "Compute the empirical esperance"
        return np.mean(self.X, axis=0)

    def variance(self, mean=None):
        """Compute the empirical variance of each dimension,
        using mean if already computed """
        if mean == None:
            mean = self.mean()
        return np.sum((mean - self.X) ** 2, axis=0) / (self.N - 1)

    def interval(self, alpha):
        """Compute the 1-alpha confidence interval of the expected value"
        return the mean and the error s.t. I=[mu +- error] """
        mu = self.mean()
        err = st.norm.ppf(1 - alpha / 2) * np.sqrt(self.variance(mu) / self.N)
        return mu, err
