#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:54:01 2023

@author: conocirone
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pandas as pd
from skimage import data

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




A = data.camera()



A_p1 = np.zeros(A.shape)
A_p2 = np.zeros(A.shape)
A_p3 = np.zeros(A.shape)
A_p4 = np.zeros(A.shape)
U, sigma, V = scipy.linalg.svd(A)


err_rel_10 = np.zeros((10))
cp_10 = np.zeros((10))
for i in range(10):
 ui = U[:,i]
 vi = V[i,:]
 A_p1 = A_p1 + (np.outer(ui, vi) * sigma[i])
 err_rel_10[i] = np.linalg.norm(A - A_p1, 2) / np.linalg.norm(A,2)
 cp_10[i] = (min(A.shape) / (i + 1)) - 1 


err_rel_80 = np.zeros((80))
cp_80 = np.zeros((80))
for i in range(80):
 ui = U[:,i]
 vi = V[i,:]
 A_p2 = A_p2 + (np.outer(ui, vi) * sigma[i])
 err_rel_80[i] = np.linalg.norm(A - A_p2, 2) / np.linalg.norm(A,2)
 cp_80[i] = (min(A.shape) / (i + 1)) - 1 

err_rel_160 = np.zeros((160))
cp_160 = np.zeros((160))
for i in range(160):
 ui = U[:,i]
 vi = V[i,:]
 A_p3 = A_p3 + (np.outer(ui, vi) * sigma[i])
 err_rel_160[i] = np.linalg.norm(A - A_p3, 2) / np.linalg.norm(A,2)
 cp_160[i] = (min(A.shape) / (i + 1)) - 1 

err_rel_340 = np.zeros((340))
cp_340 = np.zeros((340))
for i in range(340):
 ui = U[:,i]
 vi = V[i,:]
 A_p4 = A_p4 + (np.outer(ui, vi) * sigma[i])
 err_rel_340[i] = np.linalg.norm(A - A_p4, 2) / np.linalg.norm(A,2)
 cp_340[i] = (min(A.shape) / (i + 1)) - 1 


plt.figure(figsize=(15, 5))
fig1 = plt.subplot(2, 3, 1)
fig1.imshow(A, cmap='gray')
plt.title('True image')
fig2 = plt.subplot(2, 3, 2)
fig2.imshow(A_p1, cmap='gray')
plt.title('Reconstructed image with p =' + str(10))
fig3 = plt.subplot(2, 3, 3)
fig3.imshow(A_p2, cmap='gray')
plt.title('Reconstructed image with p =' + str(80))
fig4 = plt.subplot(2, 3, 4)
fig4.imshow(A_p3, cmap='gray')
fig5 = plt.subplot(2, 3, 5)
fig5.imshow(A_p4, cmap='gray')
plt.show()

print('fattore di compressione per ')

plt.figure(figsize=(10, 5))
fig1 = plt.subplot(1, 2, 1)
fig1.plot(err_rel_10, label = 'p = 10', color = 'blue')
fig1.plot(err_rel_80,label = 'p = 80', color = 'green')
fig1.plot(err_rel_160, label = 'p = 160', color = 'yellow')
fig1.plot(err_rel_340, label = 'p = 340', color = 'red')
plt.legend(loc = 'upper right')
plt.title('Errore relativo')
fig2 = plt.subplot(1, 2, 2)
fig2.plot(cp_10,label = 'p = 10', color = 'blue')
fig2.plot(cp_80, label = 'p = 80', color = 'green')
fig2.plot(cp_160, label = 'p = 160', color = 'yellow')
fig2.plot(cp_340, label = 'p = 340', color = 'red')

plt.legend(loc = 'upper right')
plt.title('Fattore di compressione')
plt.show()

print('fattore di compressione per p = 10: ', cp_10[-1])
print('fattore di compressione per p = 80: ', cp_80[-1])
print('fattore di compressione per p = 160: ', cp_160[-1])
print('fattore di compressione per p = 340: ', cp_340[-1])





