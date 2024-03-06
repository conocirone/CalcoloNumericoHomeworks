#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:50:19 2023

@author: conocirone
"""

import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec

K_T = np.zeros((290,1))
Err_T = np.zeros((290,1))

for n in range(10,300):
     #matrice tridiagonale:
     T = np.eye(n)
     c = np.diag(np.ones(n) * 9, k = 0)
     s = np.diag(np.ones(n-1)* -4, k=1)
     i = np.diag(np.ones(n-1)* -4, k=-1)
     T  = s + i + c
     x_T = np.ones((n,1))
     b_T = T@x_T
     
     #numero di condizione:
     K_T[n-10] = np.linalg.cond(T, 2)
     L_T = scipy.linalg.cholesky(T, lower = True)
     T  = np.dot(L_T, L_T.transpose())
     b_T = T@x_T
     my_x_T = T.transpose()@b_T
     
     
     #errore relativo:
     Ea_T = my_x_T - x_T
     Err_T[n-10] = scipy.linalg.norm(Ea_T, 2) / scipy.linalg.norm(x_T,2)
     
     
plt.plot(K_T)
plt.semilogy(K_T)
plt.semilogx(K_T)
plt.title('CONDIZIONAMENTO DI T ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(T)')
plt.show()

plt.plot( Err_T)
plt.semilogy(Err_T)
plt.semilogx(Err_T)
plt.title('Errore relativo di T')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err= ||my_x-x||/||x||')
plt.show()

