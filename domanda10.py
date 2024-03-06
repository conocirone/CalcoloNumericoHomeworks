#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:49:24 2023

@author: conocirone
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


def equazioni_normali(A,y):
    ATA = A.T@A
    ATy = A.T@y
    L = scipy.linalg.cholesky(ATA, lower = True)
    alpha1= scipy.linalg.solve(L,ATy)
    alpha_normali = scipy.linalg.solve(L.T, alpha1)
    return alpha_normali

def SVD(A,n, y):
    U, sigma, V = scipy.linalg.svd(A)
    alpha_svd = np.zeros(sigma.shape)
    for i in range(n+1):
        ui = U[:,i] 
        vi = V[i,:] 
        alpha_svd = alpha_svd + (np.dot(ui,y)*vi)/sigma[i]
    return alpha_svd

def p(alpha, x):
  N = x.size 
  A = np.ones((N, len(alpha)))
  for i in range (len(alpha)):
    A[:,i] = x**i
  y_plot = np.dot(A, alpha)
  return y_plot


def matrix_A(x, n, N):
 A = np.zeros((N, n+1))
 for i in range(n+1):
   A[:,i] = x**i
 return A

f = lambda x: np.sin(5*x) + 3*x
x_f = np.linspace(1, 5, 10)
y1 = f(x_f)

x_f_2 = np.linspace(1, 5, 30)
y2 = f(x_f_2)

x_f_3 = np.linspace(1, 5, 50)
y3 = f(x_f_3)

def create_subplot(x, y, label, color, marker, i):
    plt.subplot(2, 3, i)
    plt.plot(x,y,label = label, color = color, marker = marker)
    plt.legend(loc = 'upper right')
    
def comparison(n, x, y, f, i, N):
  A = matrix_A(x, n, N)  
  alpha_normali = equazioni_normali(A,y)
  alpha_svd = SVD(A,n,y)
  y_normali = p(alpha_normali,x)
  y_svd = p(alpha_svd,x)
  err1 = np.linalg.norm (y-y_normali, 2) 
  err2 = np.linalg.norm (y-y_svd, 2) 
  print ('Errore di approssimazione con Eq. Normali per n = ', n, ':',  err1)
  print ('Errore di approssimazione con SVD per n = ',n, ':', err2)
  create_subplot(x, y_normali,'eq. normali','green', 'o', i)
  create_subplot(x, y_svd,'svd', 'yellow', '*', i)
  create_subplot(x, y,'grafico f(x)', 'blue', '_', i)
   

plt.figure(figsize = (15, 8))
comparison(1, x_f, y1, f, 1, 10)
comparison(3, x_f, y1, f, 2, 10)
comparison(5, x_f, y1, f, 3, 10)
comparison(7, x_f, y1, f, 4, 10)
plt.show()


plt.figure(figsize = (15, 8))
comparison(1, x_f_2, y2, f, 1, 30)
comparison(3, x_f_2, y2, f, 2, 30)
comparison(5, x_f_2, y2, f, 3, 30)
comparison(7, x_f_2, y2, f, 4, 30)
plt.show()

plt.figure(figsize = (15, 8))
comparison(1, x_f_3, y3, f, 1, 50)
comparison(3, x_f_3, y3, f, 2, 50)
comparison(5, x_f_3, y3, f, 3, 50)
comparison(7, x_f_3, y3, f, 4, 50)
plt.show()
