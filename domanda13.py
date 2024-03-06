#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:00:20 2023

@author: conocirone
"""

import numpy as np
import math
import time
import matplotlib.pyplot as plt


def bisezione(a, b, f, tolx, xTrue):
  starting_time = time.time()
  k = math.ceil(math.log(abs(b-a)/tolx)/math.log(2))     
  vecErrore = np.zeros( (k,1) )
  if f(a)*f(b)>0:
     print("Non ci sono intersezioni sull'asse x")
     return(0, 0, 0, 0, 0)
  for i in range(1,k):
    c = a + ((b - a)/2)
    vecErrore[i-1] = abs(c - xTrue)
    if abs(f(c)) < 1.e-16:                              
          x = c
          return (x, i, k, vecErrore)
    else:
        if np.sign(f(a)*np.sign(f(c))) < 0:
         b = c
        else:
         a = c
    x=c
    time_bis = time.time() - starting_time
  vecErrore = vecErrore[0:i]
  return (x, i, k, vecErrore, time_bis)


def newton( f, df, tolf, tolx, maxit, xTrue, x0):
  starting_time = time.time()
  err=np.zeros(maxit, dtype=float)
  vecErrore=np.zeros( (maxit,1), dtype=float)
  
  
  i=0
  err[0]=tolx+1 
  vecErrore[0] = abs(x0-xTrue)
  x=x0

  while ( abs(f(x)) > tolf and err[i] > tolx  and i < maxit - 1): 
    x0 = x
    x = x0  - (f(x0) / df(x0))
    i = i+1
    err[i] = abs(x - x0)
    vecErrore[i] = abs(x - xTrue)
    
  err = err[0:i]
  vecErrore = vecErrore[0:i]
  time_newton = time.time() - starting_time
  return (x, i, err, vecErrore, time_newton)  

def succ_app(f, g, tolf, tolx, maxit, xTrue, x0):
  starting_time = time.time()
  err=np.zeros(maxit+1, dtype=np.float64)
  vecErrore=np.zeros(maxit+1, dtype=np.float64)
  
  
  i= 0
  err[0]=tolx+1
  vecErrore[0] = abs(x0-xTrue)
  x = x0

  while ( abs(f(x)) > tolf and err[i] > tolx  and i < maxit - 1 ): 
     x0 = x
     x = g(x0)
     i = i+1
     err[i] = abs(x - x0)
     vecErrore[i] = abs(x - xTrue)
    
  err = err[0:i]
  vecErrore = vecErrore[0:i]
  time_succ = time.time() - starting_time
  return (x, i, err, vecErrore, time_succ) 


f = lambda x: x**3 + 4*x*np.cos(x) - 2
df = lambda x: 3*x**2 + 4*np.cos(x) - 4*x*np.sin(x)
 
g1 = lambda x: (2-x**3)/(4*np.cos(x))


xTrue = 0.536839
a = 0
b = 2
tolx = 10**(-10)
tolf = 10**(-6)
maxit=100
x0= tolx

''' Grafico funzione in [a, b]'''
x_plot = np.linspace(a, b, 101)
[sol_bis, iter_bis, k_bis, vecErrore_bis, time_bis] = bisezione(a,b,f,tolx,xTrue)
[sol_new, iter_new, err_new, vecErrore_new, time_new] = newton(f, df, tolf, tolx, maxit, xTrue, x0)
[sol_succ1, ite_succ1, err_succ1, vecErr_succ1, time_succ1] = succ_app(f, g1, tolf, tolx, maxit, xTrue, x0)


print(sol_bis)
print(sol_new)
print(sol_succ1)

print(iter_bis)
print(iter_new)
print(ite_succ1)


print(vecErrore_bis[-1])
print(vecErrore_new[-1])
print(vecErr_succ1[-1])

''' Grafico Errore vs Iterazioni'''
ite_b = np.arange(1, iter_bis + 1)
ite_n = np.arange(1,iter_new +1)  
ite_succg1 = np.arange(1,ite_succ1+1)  
 

plt.figure(figsize = (15,5))
plt.plot(ite_b, vecErrore_bis, label = 'bisezione', color = 'yellow' )
plt.plot(ite_n, vecErrore_new,label = 'newton', color = 'blue' )
plt.plot(ite_succg1, vecErr_succ1,label = 'app succ g1', color = 'red' )
plt.xlabel('Iterazioni')
plt.ylabel('Errore')
plt.show()



