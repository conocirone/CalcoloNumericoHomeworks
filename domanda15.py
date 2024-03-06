#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 22:04:13 2023

@author: conocirone
"""

import numpy as np
import matplotlib.pyplot as plt

def next_step(x,grad): 
  alpha=1.1
  alpha_min=10**(-10)
  rho = 0.5
  c1 = 0.25
  p=-grad
  j=0
  jmax=10
  while((f(x + alpha*p)>f(x)+c1*alpha*grad.T@p) and (j<jmax) and (alpha>alpha_min)):
      alpha=rho*alpha
      j=j+1
  if (j==jmax or alpha<=alpha_min):     
      return -1    
  else:
      return alpha  
    

def minimize(x0, x_true, step, fixed_step, MAXITERATION, ABSOLUTE_STOP):
    x = np.zeros((2,MAXITERATIONS))
    norm_grad_list=np.zeros((1,MAXITERATION))
    function_eval_list=np.zeros((1,MAXITERATION))
    error_list=np.zeros((1,MAXITERATION)) 
    
    k=0
    x_last = np.array([x0[0],x0[1]])
    x[:,k] = x_last
    function_eval_list[:,k]=f(x0)
    error_list[:,k]=np.linalg.norm(x0 - x_true)
    norm_grad_list[:,k]=np.linalg.norm(grad_f(x0))
    
    while (np.linalg.norm(grad_f(x_last))>=ABSOLUTE_STOP and k < MAXITERATION - 1 ):   
        k=k+1
        grad=grad_f(x_last)
        
        if fixed_step == True:
            step_1 = step
        else :
            step_1 = next_step(x_last, grad)
        
        if(step_1 == -1):
            print("non converge")
            return
        
        x_last = x_last - step_1*grad
        
        x[:,k] = x_last
        function_eval_list[:,k] = f(x_last)
        error_list[:, k] = np.linalg.norm(x_last - x_true)
        norm_grad_list[:,k]= np.linalg.norm(grad_f(x_last))
        # print(norm_grad_list)
    
    function_eval_list = function_eval_list[:,:k]
    error_list = error_list[:, :k]
    norm_grad_list = norm_grad_list[:, :k]
    
    return (x_last, norm_grad_list,function_eval_list, error_list,k,x)




'''creazione del problema'''



def f(x):#f va da R^2 in R
  f = 10*(x[0]-1)**2 + (x[1]-2)**2
  return f

def grad_f(x):
    return np.array([ 20*x[0]-20 , 2*x[1]-4])

step=0.01
MAXITERATIONS=1000
ABSOLUTE_STOP=10**-5
x_true=np.array((1,2))
x0 = np.array((3,-5))

[x_last_var, norm_grad_list_var, function_eval_list_var, error_list_var, k_var, x_var] = minimize(x0, x_true, step, False, MAXITERATIONS, ABSOLUTE_STOP)

[x_last_fixed, norm_grad_list_fixed, function_eval_list_fixed, error_list_fixed, k_fixed, x_fixed] = minimize(x0, x_true, step, True, MAXITERATIONS, ABSOLUTE_STOP)


k_plot_var = np.arange(k_var )
norm_grad_list_plot_var = np.reshape(norm_grad_list_var[:,0:k_var],k_var)
error_list_plot_var = np.reshape(error_list_var[:,0:k_var],k_var)
function_eval_list_plot_var = np.reshape(function_eval_list_var[:,0:k_var],k_var)



k_plot_fixed = np.arange(k_fixed)
norm_grad_list_plot_fixed=np.reshape(norm_grad_list_fixed[:,0:k_fixed],k_fixed)
error_list_plot_fixed = np.reshape(error_list_fixed[:,0:k_fixed],k_fixed)
function_eval_list_plot_fixed = np.reshape(function_eval_list_fixed[:,0:k_fixed],k_fixed)


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.loglog(k_plot_var,norm_grad_list_plot_var)
plt.xlabel("Iterazioni")
plt.ylabel("Norma Gradiente")
plt.title('Norma Gradiente vs Iterazioni, step size variabile')
plt.subplot(1, 2, 2)
plt.loglog(k_plot_fixed,norm_grad_list_plot_fixed)
plt.xlabel("Iterazioni")
plt.ylabel("Norma Gradiente")
plt.title('Iterazioni vs Norma Gradiente step size fisso')
plt.show()


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.loglog(k_plot_var,error_list_plot_var)
plt.title('Errore vs Iterazioni step size variabile')
plt.xlabel("Iterazioni")
plt.ylabel("Errore")
plt.subplot(1, 2, 2)
plt.loglog(k_plot_fixed,error_list_plot_fixed)
plt.title('Errore vs Iterazioni step size fisso')
plt.xlabel("Iterazioni")
plt.ylabel("Errore")
plt.show()


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.loglog(k_plot_var,function_eval_list_plot_var)
plt.title('Funzione Obiettivo vs Iterazioni step size variabile')
plt.xlabel("Iterazioni")
plt.ylabel("Funzione Obiettivo")
plt.subplot(1, 2, 2)
plt.loglog(k_plot_fixed,function_eval_list_plot_fixed)
plt.title('Iterazioni vs Funzione Obiettivo step size fisso')
plt.xlabel("Iterazioni")
plt.ylabel("Funzione Obiettivo")
plt.show()


print(k_var)



