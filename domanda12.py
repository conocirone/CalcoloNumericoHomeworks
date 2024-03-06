#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:36:35 2023

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
  for i in range(0,k):
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


f = lambda x: np.exp(x) * x**2
df = lambda x: np.exp(x)*x**2 + np.exp(x)*2*x
 
g1 = lambda x: x - f(x)*np.exp(x/2)
g2 = lambda x: x - f(x) * np.exp((-x)/2)
g3 = lambda x: x - (f(x))/df(x)

xTrue = 0
a = -1
b = 1
tolx = 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 1

''' Grafico funzione in [a, b]'''
x_plot = np.linspace(a, b, 101)
[sol_bis, iter_bis, k_bis, vecErrore_bis, time_bis] = bisezione(a,b,f,tolx,xTrue)
[sol_new, iter_new, err_new, vecErrore_new, time_new] = newton(f, df, tolf, tolx, maxit, xTrue, x0)
[sol_succ1, ite_succ1, err_succ1, vecErr_succ1, time_succ1] = succ_app(f, g1, tolf, tolx, maxit, xTrue, x0)
[sol_succ2, ite_succ2, err_succ2, vecErr_succ2, time_succ2] = succ_app(f, g2, tolf, tolx, maxit, xTrue, x0)
[sol_succ3, ite_succ3, err_succ3, vecErr_succ3, time_succ3] = succ_app(f, g3, tolf, tolx, maxit, xTrue, x0)
print('Soluzioni calcolate dai vari metodi:')
print(sol_new)
print(sol_succ1)
print(sol_succ2)
print(sol_succ3)
print('Errore commesso dai vari metodi: ')
print(vecErrore_new[-1])
print(vecErr_succ1[-1])
print(vecErr_succ2[-1])
print(vecErr_succ3[-1])
print('Iterazioni')
print(iter_new)
print(ite_succ1)
print(ite_succ2)
print(ite_succ3)

''' Grafico Errore vs Iterazioni'''
# vecErrore_bis = vecErrore_bis[0:iter_bis]
ite_n = np.arange(1,iter_new +1)  
ite_succg1 = np.arange(1,ite_succ1+1)  
ite_succg2 = np.arange(1,ite_succ2+1)  
ite_succg3 = np.arange(1,ite_succ3+1) 

plt.figure(figsize = (10,5))
# plt.plot(iter_bis, vecErrore_bis, label = 'bisezione', color = 'yellow' )
plt.plot(ite_n, vecErrore_new,label = 'newton', color = 'blue' )
plt.plot(ite_succg1, vecErr_succ1,label = 'app succ g1', color = 'red' )
plt.plot(ite_succg2, vecErr_succ2,label = 'app succ g2', color = 'green' )
plt.plot(ite_succg3, vecErr_succ3,label = 'app succ g3', color = 'yellow')
plt.legend(loc = 'upper right')
plt.xlabel('Iterazioni')
plt.ylabel('Errore')
plt.show()










