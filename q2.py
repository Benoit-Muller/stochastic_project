#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:10:21 2021

@author: benoitmuller
question 2.
"""

import numpy as np
import scipy as sp
import scipy.stats  as st
import matplotlib.pyplot as plt
import time
import numpy.random as rnd
from mes_stats import RandomVariable
from sobol_new import *
from Payoff import Payoff

alpha=0.01
N=2**5
m=512
X=Payoff(m)
U = generate_points(N,X.m-1,0)
Mu = np.array([X.integrate(U[n,:],position=0,order=5) for n in range(N)])
X.set_data(Mu)
print(X.PIQMC1(N,alpha=0.01,order=5,position=0,K=20))