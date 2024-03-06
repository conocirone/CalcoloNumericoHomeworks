#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:57:38 2023

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

data = pd.read_csv("HeightVsWeight.csv")
data = np.array(data)

x = data[:, 0]
y = data[:, 1]

N = x.size 

for i in range(1,8):
    A = np.zeros((N, i+1))
    A = matrix_A(x, i)
    alpha_normali = equazioni_normali(A, y)
    alpha_svd = SVD(A,i,y)
    x_plot = np.linspace(10,80, num=300)
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