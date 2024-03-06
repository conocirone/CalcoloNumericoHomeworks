#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:55:40 2023

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



# Test 1
"""considerare la precedente matrice tridiagonale per N = 100"""

dim = np.arange(5, 200, 5)

ErrRelF_J = np.zeros(np.size(dim))
ErrRelF_GS = np.zeros(np.size(dim))

ite_J = np.zeros(np.size(dim))
ite_GS = np.zeros(np.size(dim))

ErrRelF_J2 = np.zeros(np.size(dim))
ErrRelF_GS2 = np.zeros(np.size(dim))

ite_J2 = np.zeros(np.size(dim))
ite_GS2 = np.zeros(np.size(dim))

ErrRelF_J3 = np.zeros(np.size(dim))
ErrRelF_GS3 = np.zeros(np.size(dim))

ite_J3 = np.zeros(np.size(dim))
ite_GS3 = np.zeros(np.size(dim))


def comparison1(tol, ite_Ja, ErrRel_Ja, ite_GSa, ErrRel_Gsa):
 i = 0
 for n in dim:
    
    #crezione problema test
    A = np.eye(n)
    c = np.diag(np.ones(n) * 9, k = 0)
    s = np.diag(np.ones(n-1)*-4, k=1)
    k = np.diag(np.ones(n-1)*-4, k=-1)
    A = s + k + c
    xTrue = np.ones((n,1))
    b = A@xTrue
    # metodi iterativi
    x0 = np.zeros((n,1))
    maxit = 500
    tol = tol
    (xJacobi1, kJacobi1, relErrJacobi1, errIterJacobi1,timej1) = Jacobi(A, b, x0, maxit, tol, xTrue ) 
    (xGS1, kGS1, relErrGS1, errIterGS1,timegs1) = GaussSeidel(A, b, x0, maxit, tol, xTrue ) 
    #errore relativo finale
    ErrRel_Ja[i] =  np.linalg.norm(xTrue-xJacobi1)/np.linalg.norm(xTrue)
    ErrRel_Gsa[i] = np.linalg.norm(xTrue-xGS1)/np.linalg.norm(xTrue)  
    # iterazioni
    ite_Ja[i] = kJacobi1
    ite_GSa[i] = kGS1
    
    i=i+1

def grafico_iterazioni_J_GS(kJacobi, kGS, tol, i):
    plt.subplot(1,3,i)
    plt.semilogy(dim, kJacobi,label='Jacobi', color='blue', linewidth=1 )
    plt.semilogy(dim, kGS , label='Gauss Seidel', color = 'red', linewidth=2 )
    plt.legend(loc='upper right')
    plt.title('Metodo di Jacobi e GS  tol = ' + str(tol))
    plt.xlabel('dimensione matrice: N' )
    plt.ylabel('numero iterazioni')
    
def grafico_errore_J_GS(ErrRel1, ErrRel2, tol, i):
    plt.subplot(1,3,i)
    plt.semilogy(dim, ErrRel1,label='Jacobi', color='blue', linewidth=1 )
    plt.semilogy(dim, ErrRel2 , label='Gauss Seidel', color = 'red', linewidth=2, )
    plt.legend(loc='upper right')
    plt.title('Metodo di Jacobi con e tol = ' + str(tol))
    plt.xlabel('dimensione matrice: N' )
    plt.ylabel('errore relativo')




ErrRelF_J_pi = np.zeros(80)
ErrRelF_GS_pi = np.zeros(80)

def variare_punto_iniziale(valori_da_variare, tol, ErrRel_Ja,  ErrRel_Gsa):
    
    for i in range(0,80):
     dim = 200
     valore_iniziale = np.full((dim,1), valori_da_variare[i])
     A = np.eye(dim)
     c = np.diag(np.ones(dim) * 9, k = 0)
     s = np.diag(np.ones(dim-1)*-4, k=1)
     k = np.diag(np.ones(dim-1)*-4, k=-1)
     A = s + k + c
     xTrue = np.ones((dim,1))
     b = A@xTrue
     # metodi iterativi
     x0 = valore_iniziale 
     maxit = 500
     tol = tol
     (xJacobi1, kJacobi1, relErrJacobi1, errIterJacobi1,timej1) = Jacobi(A, b, x0, maxit, tol, xTrue ) 
     (xGS1, kGS1, relErrGS1, errIterGS1,timegs1) = GaussSeidel(A, b, x0, maxit, tol, xTrue ) 
     #errore relativo finale
     ErrRelF_J_pi[i] =  np.linalg.norm(xTrue-xJacobi1)/np.linalg.norm(xTrue)
     ErrRelF_GS_pi[i] = np.linalg.norm(xTrue-xGS1)/np.linalg.norm(xTrue)   
     
valori_da_variare = np.linspace(0.1,5, 80 )
variare_punto_iniziale(valori_da_variare, 1.e-7, ErrRelF_J_pi, ErrRelF_GS_pi)
plt.figure(figsize = (10,5))
plt.loglog(valori_da_variare, ErrRelF_J_pi, color = 'green', label = 'Jacobi')
plt.loglog(valori_da_variare, ErrRelF_GS_pi, color = 'blue', label = 'GS')
plt.title('Errore al variare di x0')
plt.legend(loc = 'upper right')
plt.xlabel('punti iniziali')
plt.ylabel('errore relativo')
plt.show()



tol = [1.e-5, 1.e-7, 1.e-9]

comparison1( tol[0], ite_J, ErrRelF_J, ite_GS, ErrRelF_GS)
comparison1( tol[1], ite_J2, ErrRelF_J2, ite_GS2, ErrRelF_GS2)
comparison1( tol[2], ite_J3, ErrRelF_J3, ite_GS3, ErrRelF_GS3)

plt.figure(figsize = (20,5))
grafico_iterazioni_J_GS( ite_J ,ite_GS, tol[0], 1)
grafico_iterazioni_J_GS( ite_J2 ,ite_GS2, tol[1], 2)
grafico_iterazioni_J_GS( ite_J2 , ite_GS3, tol[2], 3)
plt.show()

plt.figure(figsize = (20,5))
grafico_errore_J_GS( ErrRelF_J,ErrRelF_GS,  tol[0], 1)
grafico_errore_J_GS( ErrRelF_J2 , ErrRelF_GS2, tol[1], 2)
grafico_errore_J_GS( ErrRelF_J3,ErrRelF_GS3, tol[2], 3)
plt.show()


