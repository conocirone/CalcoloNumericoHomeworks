#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:32:52 2023

@author: conocirone
"""

import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec

K_H = np.zeros((13,1))
Err_H= np.zeros((13,1))


for n in np.arange(2,15):
    #creazione problema test
    #matrice hilbert:
    H = scipy.linalg.hilbert(n)
    x = np.ones((n,1))
    b = H@x
    
    #numero di condizione:
    K_H[n-2] = np.linalg.cond(H, 2)
    
    #fattorizzazione choleski:
    L = scipy.linalg.cholesky(H, lower = True)
    H = np.dot(L, L.transpose())
    b = H@x
    my_x = H.transpose()@b
    
    #errore relativo:
    Ea = my_x - x
    Err_H[n-2] = scipy.linalg.norm(Ea, 2) / scipy.linalg.norm(x,2)
    
plt.plot(K_H)
plt.semilogy(K_H)
plt.semilogx(K_H)
plt.title('CONDIZIONAMENTO DI MATRICE DI HILBERT ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()

plt.plot(Err_H)
plt.semilogy(Err_H)
plt.semilogx(Err_H)
plt.title('Errore relativo matrice di Hiblbert:')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err= ||my_x-x||/||x||')
plt.show()

