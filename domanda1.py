#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:27:37 2023

@author: conocirone
"""
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec



K_A = np.zeros((990,1))
Err = np.zeros((990,1))
for n in range(10, 1000): 
    #problema test
    A = np.matlib.rand((n, n))
    x = np.ones((n,1))
    b = A@x
    
    #numero di condizione:
    K_A[n-10] = np.linalg.cond(A, 2)
    
    
    #fattorizzazione:
    lu, piv = LUdec.lu_factor(A)
    my_x=scipy.linalg.lu_solve((lu, piv), b)
    
    #errore relativo:
    Ea = my_x - x
    Err[n-10] =scipy.linalg.norm(Ea, 2) / scipy.linalg.norm(x,2)
    


plt.plot(K_A)
plt.semilogy(K_A)
plt.semilogx(K_A)
plt.title('CONDIZIONAMENTO DI A ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()


plt.plot(Err)
plt.semilogy(Err)
plt.semilogx(Err)
plt.title('Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err= ||my_x-x||/||x||')
plt.show()

