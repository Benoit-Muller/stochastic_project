#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:28:31 2021

@author: benoitmuller
––––––––––––––––––––––––––––––––––
            PROJECT 5
––––––––––––––––––––––––––––––––––
"""
#ceci est un test pour essayer github
import numpy as np
import scipy as sp
import scipy.stats  as st
import matplotlib.pyplot as plt
import time
import numpy.random as rnd
from mes_stats import RandomVariable
from sobol_new import *

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
    res=0
    for i in range(len(w)):
        res=res+S(t[i],w[i])
    return max(res/len(w)-K,0)
def Psi2(t,w):
    return Psi1(t,w)>0
def normal(U):
    m=len(U)
    if m%2==1:
        print("Length of sample must be even!") #to be removed
    rho = np.sqrt(-2 * np.log(U[:int(m/2)]))
    theta = 2 * np.pi * U[int(m/2):]
    Z=np.zeros(m)
    Z[:int(m/2)]  = rho * np.cos(theta)
    Z[int(m/2):]  = rho * np.sin(theta)
    return Z
def weiner(t,U):
    Z=normal(U)
    W=np.zeros(len(Z))
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
def QMC(transform,d=1,N=100,alpha=0.1):
    U = generate_points(N,d,0)
    X=np.zeros(N)
    for n in range(N):
        X[n]= transform(U[n,:])
    X=RandomVariable(X)
    return X.interval(alpha)
"""
#%% Tests
m=10
N=100
NN=np.arange(1,N+1)
U=rnd.uniform(size=m)
Z=RandomVariable(normal(U))
t=np.linspace(0,1,m)
W=weiner(t,U)
gen=lambda : Psi1(t,weiner(t,rnd.uniform(size=m)))
mu,err = np.zeros(N),np.zeros(N)
for n in range(N):
    mu[n],err[n] = MC(gen,n)
plt.figure()
plt.plot(mu)
plt.plot(mu+err)
plt.plot(mu-err)
"""
# Goal 1.
NN=2**np.arange(7,14) #(7,14)
mm=2**np.arange(5,6) #(5,10)
alpha=0.1

mu1MC = np.zeros(len(NN))
err1MC = np.zeros(len(NN))
for m in mm:
    dt=T/m
    t=dt * np.arange(1,m+1)
    Z1gen = lambda : Psi1(t,weiner(t,rnd.uniform(size=m)))
    Z2trans = lambda U: Psi2(t,weiner(t,U))
    for j in range(len(NN)):
        mu1MC[j], err1MC[j] = MC(Z1gen,NN[j],alpha)
        mu1QMC[j], err1QMC[j] = QMC(Z1gen,NN[j],alpha)
    plt.figure()
    plt.xscale('log')
    plt.fill_between(NN,mu1MC+err1MC,mu1MC-err1MC,alpha=0.5)
    plt.plot(NN,mu1MC)
    plt.title("Goal 1\n"+
              "Crude Monte Carlo with m="+str(m)+', confidence 1-'+str(alpha))
    plt.xlabel('Sample size N')
    plt.ylabel('mean and confidence interval')
    plt.figure()
    plt.loglog(NN,err1MC,'.-',label='Estimated error')
    plt.loglog(NN,NN**(-0.5),label='$1/N^2$')
    plt.title("Goal 1 \n" + 
              " Estimated error of Crude Monte Carlo,\n" +
              "$m=" + str(m) + '$, confidence $1-'+str(alpha)+"$")
    plt.xlabel('Sample size N')
    plt.ylabel('Estimated error')
    plt.legend()
    plt.show()