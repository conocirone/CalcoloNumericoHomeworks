#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:56:22 2023

@author: conocirone
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pandas as pd


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


def matrix_A(x, n):
 A = np.zeros((N, n+1))
 for i in range(n+1):
   A[:,i] = x**i
 return A


x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])
N = x.size # Numero dei dati

for i in range(1,8):
    A = np.zeros((N, i+1))
    A = matrix_A(x, i)
    alpha_normali = equazioni_normali(A, y)
    alpha_svd = SVD(A,i,y)
    var = 100
    x_plot = np.linspace(1,3,var)
    y_normali = p(alpha_normali, x_plot)
    y_svd = p(alpha_svd, x_plot)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x,y,'o')
    plt.plot(x_plot, y_normali, color = 'blue')
    plt.title('Approssimazione tramite Eq. Normali ')
    plt.subplot(1, 2, 2)
    plt.plot(x,y,'o')
    plt.plot(x_plot, y_svd,  color ='red')
    plt.title('Approssimazione tramite SVD')



