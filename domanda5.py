#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:51:40 2023

@author: conocirone
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.linalg.decomp_lu as LUdec 
import time


"""metodo di Jacobi"""

def Jacobi(A,b,x0,maxit,tol, xTrue):
  n=np.size(x0)     
  ite=0 
  x = np.copy(x0) 
  norma_it=1+tol 
  relErr=np.zeros((maxit, 1)) 
  errIter=np.zeros((maxit, 1)) 
  relErr[0]=np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue) 
  while (ite<maxit - 1 and norma_it>tol):
    x_old=np.copy(x)
    for i in range(0,n):
     
      x[i]=(b[i]-np.dot(A[i,0:i],x_old[0:i])-np.dot(A[i,i+1:n],x_old[i+1:n]))/A[i,i] 
    
    ite=ite+1
    norma_it = np.linalg.norm(x_old-x)/np.linalg.norm(x_old)  
    relErr[ite] = np.linalg.norm(xTrue-x)/np.linalg.norm(xTrue) 
    errIter[ite-1] = norma_it
  relErr=relErr[:ite]
  errIter=errIter[:ite]  
  time_J = time.time()
  return [x, ite, relErr, errIter, time_J]

"""metodo di Gauss Seidel"""

def GaussSeidel(A,b,x0,maxit,tol, xTrue):
 n = np.size(x0)
 ite = 0
 x = np.copy(x0)
 relErr = np.zeros((maxit, 1))
 errIter = np.zeros((maxit, 1))
 errIter[0] = tol+1
 relErr[0] = np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue)
 while( ite < maxit - 1 and errIter[ite] > tol):
    x_old = np.copy(x)
    for i in range (0,n):
        x[i] = (b[i] - np.dot(A[i, 0:i], x[0:i]) - np.dot(A[i, i+1:n], x_old[i+1:n]))/A[i,i]
    ite = ite + 1
    relErr[ite] = np.linalg.norm(xTrue-x)/np.linalg.norm(xTrue)
    errIter[ite] = np.linalg.norm(x-x_old)/np.linalg.norm(x)
 relErr = relErr[:ite]
 errIter = errIter[:ite]
 time_GS = time.time()
 return [x, ite, relErr, errIter, time_GS]


def LU(A, b):
    lu, piv = LUdec.lu_factor(A)
    my_x=scipy.linalg.lu_solve((lu, piv), b)
    timing = time.time()
    return [timing, my_x]

def Cholesky(A, b):
    L = scipy.linalg.cholesky(A, lower = True)
    y = scipy.linalg.solve(L, b)
    my_x = scipy.linalg.solve(L.T, y)
    timing = time.time()
    return [timing, my_x]

dim = np.arange(5, 100, 5)


time_Jacobi = np.zeros(np.size(dim))
time_GaussS = np.zeros(np.size(dim))
time_LU = np.zeros(np.size(dim))
time_Cholesky = np.zeros(np.size(dim))

ErrRelF_J = np.zeros(np.size(dim))
ErrRelF_GS = np.zeros(np.size(dim))
ErrRel_LU = np.zeros(np.size(dim))
ErrRel_Cho = np.zeros(np.size(dim))

i = 0

for n in dim:
    #creazione problema test:
    A = np.eye(n)
    c = np.diag(np.ones(n) * 9, k = 0)
    s = np.diag(np.ones(n-1)*-4, k=1)
    k = np.diag(np.ones(n-1)*-4, k=-1)
    A = s + k + c
    xTrue = np.ones((n,1))
    b = A@xTrue
   
    #metodi diretti:
    ErrRel_LU[i] = np.linalg.norm(xTrue-LU(A,b)[1])/np.linalg.norm(xTrue)
    ErrRel_Cho[i] = np.linalg.norm(xTrue-Cholesky(A,b)[1])/np.linalg.norm(xTrue)
    starting_time = time.time()
    time_LU[i] =  LU(A,b)[0] - starting_time
    starting_time = time.time()
    time_Cholesky[i] = Cholesky(A, b)[0] - starting_time
    
    #metodi iterativi:
    x0 = np.zeros((n,1)) #x0 non può avere norma 0, quindi setta come primo elemento 1
    x0[0] = 1
    maxit = 500
    tol = 1.e-7
    (xJacobi, kJacobi, relErrJacobi, errIterJacobi,timej) = Jacobi(A, b, x0, maxit, tol, xTrue ) 
    (xGS, kGS, relErrGS, errIterGS,timegs) = GaussSeidel(A, b, x0, maxit, tol, xTrue ) 
    ErrRelF_J[i] = np.linalg.norm(xTrue-xJacobi)/np.linalg.norm(xTrue)  
    ErrRelF_GS[i] = np.linalg.norm(xTrue-xGS)/np.linalg.norm(xTrue) 
    starting_time = time.time()
    time_Jacobi[i] = Jacobi(A, b, x0, maxit, tol, xTrue )[4] - starting_time
    starting_time = time.time()
    time_GaussS[i] = GaussSeidel(A, b, x0, maxit, tol, xTrue)[4] - starting_time
    
    i = i+1

plt.figure()
plt.semilogy(dim, time_Jacobi,label='Jacobi', color='blue', linewidth=1, marker='o'  )
plt.semilogy(dim, time_GaussS, label='Gauss Seidel', color = 'red', linewidth=2, marker='.' )
plt.semilogy(dim, time_LU,label='LU', color='green', linewidth=1, marker='o'  )
plt.semilogy(dim, time_Cholesky, label='Cholesky', color = 'yellow', linewidth=2, marker='.' )
plt.legend(loc='upper right')
plt.xlabel('DIMENSION')
plt.ylabel('Time')
plt.title('Comparison of the different algorithms')
plt.show()

plt.figure()
plt.semilogy(dim, ErrRelF_J,label='Jacobi', color='blue', linewidth=1, marker='o'  )
plt.semilogy(dim, ErrRelF_GS , label='Gauss Seidel', color = 'red', linewidth=2, marker='.' )
plt.semilogy(dim, ErrRel_LU,label='LU', color='green', linewidth=1, marker='o'  )
plt.semilogy(dim, ErrRel_Cho, label='Cholesky', color = 'yellow', linewidth=2, marker='.' )
plt.legend(loc='upper right')
plt.xlabel('DIMENSION')
plt.ylabel('Relative Error')
plt.title('Comparison of the different algorithms')
plt.show()

    