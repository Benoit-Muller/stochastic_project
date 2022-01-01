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

# File saving options:
save_figures = False # Change the value if needed
if (save_figures==True and 
    input("Do you really want to save figures"+
          " into files?\n(yes/no): ") == "no"):
    save_figures = False
    
alpha=0.01
NN=(2**np.arange(7,10)).astype(int) #(7,14)
mm=(2**np.arange(5,6)).astype(int) #(5,10)
MC1 = np.zeros((2,len(NN)))
MC2 = np.zeros((2,len(NN)))
QMC1 = np.zeros((2,len(NN)))
QMC2 = np.zeros((2,len(NN)))

plt.figure(figsize=(10, 9)) # figsize=(6,4) by default
plt.suptitle("Goal 2: With pre-integration, estimated error" +
             " with confidence $1-" + str(alpha) + "$")
for m in mm:
    X=Payoff(m)
    for j in range(len(NN)):
        MC1[:,j], MC2[:,j] = X.PIMC(NN[j],alpha,order=2)
        QMC1[:,j], QMC2[:,j] = X.PIQMC(NN[j],alpha,order=2)
    plt.subplot(221)
    plt.loglog(NN,MC1[1,:],'.-',label='$m=$'+str(m))
    plt.subplot(222)
    plt.loglog(NN,QMC1[1,:],'.-',label='$m=$'+str(m))
    plt.subplot(223)
    plt.loglog(NN,MC2[1,:],'.-',label='$m=$'+str(m))
    plt.subplot(224)
    plt.loglog(NN,QMC2[1,:],'.-',label='$m=$'+str(m))
    
# Finest result (biggest m and N):
print("The computed intervals for m =", m,
      ", N =", NN[-1],"and alpha =",alpha,"are:")
print("V1:  PIMC:",MC1[0,-1],"±",MC1[1,-1])
print("    PIQMC:",QMC1[0,-1],"±",QMC1[1,-1])
print("V2:  PIMC:",MC2[0,-1],"±",MC2[1,-1])
print("    PIQMC:",QMC2[0,-1],"±",QMC2[1,-1])

# Error plots:
plt.subplot(221)
plt.loglog(NN,NN**(-0.5),"--",label='$1/\sqrt{N}$')
plt.title("Crude Monte Carlo for V1")
plt.xlabel('Sample size N')
plt.ylabel('Estimated error')
plt.legend()

plt.subplot(222)
plt.loglog(NN,NN**(-0.5),"--",label='$1/\sqrt{N}$')
plt.loglog(NN,10/NN,"--",label='$1/N$')
plt.title("Quasi Monte Carlo for V1")
plt.xlabel('Sample size N')
plt.ylabel('Estimated error')
plt.legend()

plt.subplot(223)
plt.loglog(NN,NN**(-0.5),"--",label='$1/\sqrt{N}$')
plt.title("Crude Monte Carlo for V2")
plt.xlabel('Sample size N')
plt.ylabel('Estimated error')
plt.legend()

plt.subplot(224)
plt.loglog(NN,NN**(-0.5),"--",label='$1/\sqrt{N}$')
plt.loglog(NN,10*1/NN,"--",label='$1/N$')
plt.title("Quasi Monte Carlo for V2")
plt.xlabel('Sample size N')
plt.ylabel('Estimated error')
plt.legend()

plt.tight_layout()
if save_figures == True:
    plt.savefig('graphics/q2error.pdf')

# The interval plot:    
plt.figure(figsize=(10,9))
plt.suptitle("Goal 1: With pre-integration, interval of confidence $1-"
             + str(alpha) + "$, m="+str(mm[-1]))
plt.subplot(221)
plt.xscale('log')
plt.fill_between(NN,+MC1[0,:]+MC1[1,:],MC1[0,:]-MC1[1,:],alpha=0.5)
plt.plot(NN,MC1[0,:],'.-')
plt.title("Crude Monte Carlo for V1") 
plt.xlabel('Sample size N')
plt.ylabel('mean and confidence interval')

plt.subplot(222)
plt.xscale('log')
plt.fill_between(NN,QMC1[0,:]+QMC1[1,:],QMC1[0,:]-QMC1[1,:],alpha=0.5)
plt.plot(NN,QMC1[0,:],'.-')
plt.title("Quasi Monte Carlo for V1") 
plt.xlabel('Sample size N')
plt.ylabel('mean and confidence interval')

plt.subplot(223)
plt.xscale('log')
plt.fill_between(NN,+MC2[0,:]+MC2[1,:],MC2[0,:]-MC2[1,:],alpha=0.5)
plt.plot(NN,MC2[0,:],'.-')
plt.title("Crude Monte Carlo for V2") 
plt.xlabel('Sample size N')
plt.ylabel('mean and confidence interval')

plt.subplot(224)
plt.xscale('log')
plt.fill_between(NN,QMC2[0,:]+QMC2[1,:],QMC2[0,:]-QMC2[1,:],alpha=0.5)
plt.plot(NN,QMC2[0,:],'.-')
plt.title("Quasi Monte Carlo for V2") 
plt.xlabel('Sample size N')
plt.ylabel('mean and confidence interval')

plt.tight_layout()

if save_figures == True:
    plt.savefig('graphics/q2interval.pdf')