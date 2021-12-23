#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:28:31 2021

@author: benoitmuller
––––––––––––––––––––––––––––––––––
            PROJECT 5
––––––––––––––––––––––––––––––––––
"""

import numpy as np
import scipy as sp
import scipy.stats  as st
import matplotlib.pyplot as plt
import time
import numpy.random as rnd
from mes_stats import RandomVariable
from sobol_new import *

# File saving options:
save_figures = False # Change the value if needed
if (save_figures==True and 
    input("Do you really want to save figures into files?\n(yes/no): ")=="no"):
    save_figures = False
    
# Define constants of the problem:
K=100 
S0=100
r=0.1
sigma=0.1
T=1

# Define functions of the problem, to reduce to a uniform variable:
def S(t,w):
    return S0*np.exp((r - sigma**2/2)*t + sigma*t*w)
def Psi1(t,w):
    somme=0
    for i in range(len(w)):
        somme=somme+S(t[i],w[i])
    return max(somme/len(w)-K,0)
def Psi2(t,w):
    return Psi1(t,w)>0
def normal(U):
    """" Transform uniform rvs into normal rvs;
    dimension must be even! """
    m=len(U)
    if m%2==1:
        raise Exception("The dimension m must be even")
    U = U + 1*(U==0)
    rho = np.sqrt(-2 * np.log(U[:int(m/2)]))
    theta = 2 * np.pi * U[int(m/2):]
    Z=np.zeros(np.shape(U))
    Z[:int(m/2)]  = rho * np.cos(theta)
    Z[int(m/2):]  = rho * np.sin(theta)
    return Z
def weiner(t,U):
    Z=normal(U)
    W=np.zeros(np.shape(Z))
    W[0]=t[0]*Z[0]
    for i in range(1,len(Z)):
        W[i] = W[i-1] + (t[i] - t[i-1])*Z[i]
    return W
        
# Define the methods:
def MC(generator,N=100,alpha=0.1):
    X=np.zeros(N)
    for n in range(N):
        X[n]= generator()
    X=RandomVariable(X)
    return X.interval(alpha)
def QMC(transform,d,N=100,alpha=0.1):
    U = generate_points(N,d,0)
    X=np.zeros(N)
    for n in range(N):
        X[n]= transform(U[n,:])
    X=RandomVariable(X)
    return X.interval(alpha)

# Question 1.
NN=(2**np.arange(7,14)).astype(int) #(7,14)
mm=(2**np.arange(5,10)).astype(int) #(5,10)
alpha=0.1

mu1MC = np.zeros(len(NN))
err1MC = np.zeros(len(NN))
mu1QMC = np.zeros(len(NN))
err1QMC = np.zeros(len(NN))
plt.figure(figsize=(10, 5)) # figsize=(6,4) by default
plt.suptitle("Goal 1: Estimated error" +
             " with confidence $1-" + str(alpha) + "$")
for m in mm:
    dt=T/m
    t=dt * np.arange(1,m+1)
    Z1gen = lambda : Psi1(t,weiner(t,rnd.uniform(size=m)))
    Z1trans = lambda U: Psi1(t,weiner(t,U))
    for j in range(len(NN)):
        mu1MC[j], err1MC[j] = MC(Z1gen,NN[j],alpha)
        mu1QMC[j], err1QMC[j] = QMC(Z1trans,m,NN[j],alpha)
    plt.subplot(121)
    plt.loglog(NN,err1MC,'.-',label='$m=$'+str(m))
    plt.subplot(122)
    plt.loglog(NN,err1QMC,'.-',label='$m=$'+str(m))

# Best result for biggest m:
print("The computed means for m =", m,
      "and N =", NN[-1], "are:")
print("    MC:",mu1MC[-1])
print("    QMC:",mu1QMC[-1])

# Error plots:
plt.subplot(121)
plt.loglog(NN,NN**(-0.5),label='$1/N^2$')
plt.title("Crude Monte Carlo")
plt.xlabel('Sample size N')
plt.ylabel('Estimated error')
plt.legend()

plt.subplot(122)
plt.loglog(NN,NN**(-0.5),label='$1/N^2$')
plt.title("Quasi Monte Carlo")
plt.xlabel('Sample size N')
plt.ylabel('Estimated error')
plt.legend()
plt.tight_layout()

if save_figures == True:
    plt.savefig('graphics/q1interval.pdf')

# The interval plot:    
plt.figure(figsize=(10,5))
plt.suptitle("Goal 1: Interval of confidence $1-"
             + str(alpha) + "$, m="+str(mm[-1]))
plt.subplot(121)
plt.xscale('log')
plt.fill_between(NN,mu1MC+err1MC,mu1MC-err1MC,alpha=0.5)
plt.plot(NN,mu1MC,'.-')
plt.title("Crude Monte Carlo") 
plt.xlabel('Sample size N')
plt.ylabel('mean and confidence interval')

plt.subplot(122)
plt.xscale('log')
plt.fill_between(NN,mu1QMC+err1QMC,mu1QMC-err1QMC,alpha=0.5)
plt.plot(NN,mu1QMC,'.-')
plt.title("Quasi Monte Carlo") 
plt.xlabel('Sample size N')
plt.ylabel('mean and confidence interval')
plt.tight_layout()

if save_figures == True:
    plt.savefig('graphics/q1error.pdf')

#%% Alternative with the class Payoff:
import numpy as np
import scipy as sp
import scipy.stats  as st
import matplotlib.pyplot as plt
import time
import numpy.random as rnd
from mes_stats import RandomVariable
from sobol_new import *
from Payoff import Payoff

# File saving options:
save_figures = False # Change the value if needed
if (save_figures==True and 
    input("Do you really want to save figures into files?\n(yes/no): ")=="no"):
    save_figures = False
    
alpha=0.1
NN=(2**np.arange(7,12)).astype(int) #(7,14)
mm=(2**np.arange(5,7)).astype(int) #(5,10)
mu1MC = np.zeros(len(NN))
err1MC = np.zeros(len(NN))
mu1QMC = np.zeros(len(NN))
err1QMC = np.zeros(len(NN))

plt.figure(figsize=(10, 5)) # figsize=(6,4) by default
plt.suptitle("Goal 1: Estimated error" +
             " with confidence $1-" + str(alpha) + "$")
for m in mm:
    X=Payoff(m)
    for j in range(len(NN)):
        mu1MC[j], err1MC[j] = X.MC1(NN[j],alpha)
        mu1QMC[j], err1QMC[j] = X.QMC1(NN[j],alpha)
    plt.subplot(121)
    plt.loglog(NN,err1MC,'.-',label='$m=$'+str(m))
    plt.subplot(122)
    plt.loglog(NN,err1QMC,'.-',label='$m=$'+str(m))
    
# Finest result (biggest m and N):
print("The computed intervals for m =", m,
      "and N =", NN[-1], "are:")
print("     MC:",mu1MC[-1],"+-",err1MC[-1])
print("    QMC:",mu1QMC[-1],"+-",err1QMC[-1])

# Error plots:
plt.subplot(121)
plt.loglog(NN,NN**(-0.5),label='$1/N^2$')
plt.title("Crude Monte Carlo")
plt.xlabel('Sample size N')
plt.ylabel('Estimated error')
plt.legend()

plt.subplot(122)
plt.loglog(NN,NN**(-0.5),label='$1/N^2$')
plt.title("Quasi Monte Carlo")
plt.xlabel('Sample size N')
plt.ylabel('Estimated error')
plt.legend()
plt.tight_layout()

# The interval plot:    
plt.figure(figsize=(10,5))
plt.suptitle("Goal 1: Interval of confidence $1-"
             + str(alpha) + "$, m="+str(mm[-1]))
plt.subplot(121)
plt.xscale('log')
plt.fill_between(NN,mu1MC+err1MC,mu1MC-err1MC,alpha=0.5)
plt.plot(NN,mu1MC,'.-')
plt.title("Crude Monte Carlo") 
plt.xlabel('Sample size N')
plt.ylabel('mean and confidence interval')

plt.subplot(122)
plt.xscale('log')
plt.fill_between(NN,mu1QMC+err1QMC,mu1QMC-err1QMC,alpha=0.5)
plt.plot(NN,mu1QMC,'.-')
plt.title("Quasi Monte Carlo") 
plt.xlabel('Sample size N')
plt.ylabel('mean and confidence interval')
plt.tight_layout()

if save_figures == True:
    plt.savefig('graphics/q1interval.pdf')
    
# The interval plot:    
plt.figure(figsize=(10,5))
plt.suptitle("Goal 1: Interval of confidence $1-"
             + str(alpha) + "$, m="+str(mm[-1]))
plt.subplot(121)
plt.xscale('log')
plt.fill_between(NN,mu1MC+err1MC,mu1MC-err1MC,alpha=0.5)
plt.plot(NN,mu1MC,'.-')
plt.title("Crude Monte Carlo") 
plt.xlabel('Sample size N')
plt.ylabel('mean and confidence interval')

plt.subplot(122)
plt.xscale('log')
plt.fill_between(NN,mu1QMC+err1QMC,mu1QMC-err1QMC,alpha=0.5)
plt.plot(NN,mu1QMC,'.-')
plt.title("Quasi Monte Carlo") 
plt.xlabel('Sample size N')
plt.ylabel('mean and confidence interval')
plt.tight_layout()

if save_figures == True:
    plt.savefig('graphics/q1error.pdf')