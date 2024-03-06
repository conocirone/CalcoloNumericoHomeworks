#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 12:32:26 2023

@author: conocirone
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics
from numpy import fft
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

np.random.seed(0)



def gaussian_kernel(kernlen, sigma):
    x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)    
    kern1d = np.exp(- 0.5 * (x**2 / sigma))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def psf_fft(K, d, shape):
    
    K_p = np.zeros(shape)
    K_p[:d, :d] = K

   
    p = d // 2
    K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)

   
    K_otf = fft.fft2(K_pr)
    return K_otf


def A(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(K * x))


def AT(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(np.conj(K) * x))


def f(x): 
    x_r = np.reshape(x, (m, n))
   
    res = (0.5)*( np.sum( np.square( (A(x_r, K) -b ) )))
    return res

def df(x):
    x_r = np.reshape(x, (m, n))
    res = AT(A(x_r, K), K) -AT(b, K)
    res = np.reshape(res, m*n)   
    return res

def f1(x, lambda_): 
    x_r = np.reshape(x, (m, n))
    
    res = (0.5)*( np.sum( np.square( (A(x_r, K) -b ) ))) + (0.5)*lambda_ * np.sum(np.square(x_r))
    return res

def df1(x, lambda_):
    x_r = np.reshape(x, (m, n))
    res = AT(A(x_r, K), K) -AT(b, K) + lambda_ * x_r
    res = np.reshape(res, m*n)   
    return res

img = data.camera()
X = img.astype(np.float64) / 255.0
m, n = X.shape


K = psf_fft(gaussian_kernel(24, 3), 24, X.shape)


sigma =  0.02
noise = np.random.normal(0, sigma, size = X.shape)


b = A(X, K) + noise

PSNR = metrics.peak_signal_noise_ratio(X, b)
print('PSNR = ', PSNR)
PSNR = round(PSNR, 4)
MSE = mean_squared_error(X, b)
print('MSE = ', MSE)
MSE = round(MSE, 4)



plt.figure(figsize = (20,10))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(X, cmap = 'gray')
plt.suptitle('sigma = 0.02')
plt.title('immagine Originale')

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(b, cmap = 'gray')
plt.title(f'immagine Corrotta, PSNR : ' + str(PSNR) + '  MSE: ' + str(MSE))
plt.show()


sigma =  0.01
noise = np.random.normal(0, sigma, size = X.shape)


b = A(X, K) + noise

PSNR = metrics.peak_signal_noise_ratio(X, b)
print('PSNR = ', PSNR)
PSNR = round(PSNR, 4)
MSE = mean_squared_error(X, b)
print('MSE = ', MSE)
MSE = round(MSE, 4)



plt.figure(figsize = (20,10))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(X, cmap = 'gray')
plt.suptitle('sigma = 0.02')
plt.title('immagine Originale')

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(b, cmap = 'gray')
plt.title(f'immagine Corrotta, PSNR : ' + str(PSNR) + '  MSE: ' + str(MSE))
plt.show()




def f(x): 
    x_r = np.reshape(x, (m, n))
   
    res = (0.5)*( np.sum( np.square( (A(x_r, K) -b ) )))
    return res

def df(x):
    x_r = np.reshape(x, (m, n))
    res = AT(A(x_r, K), K) -AT(b, K)
    res = np.reshape(res, m*n)   
    return res



b = A(X, K) + noise
x0 = b
max_it_1 = 2
max_it_2 = 5
max_it_3 = 8

ite = [2, 5, 8]
PSNR_n = np.zeros(3)
MSE_n = np.zeros(3)




def f1(x, lambda_): 
    x_r = np.reshape(x, (m, n))
    
    res = (0.5)*( np.sum( np.square( (A(x_r, K) -b ) ))) + (0.5)*lambda_ * np.sum(np.square(x_r))
    return res

def df1(x, lambda_):
    x_r = np.reshape(x, (m, n))
    res = AT(A(x_r, K), K) -AT(b, K) + lambda_ * x_r
    res = np.reshape(res, m*n)   
    return res

res_n_1 = minimize(f, x0, method='CG', jac=df, options={'maxiter':max_it_1, 'return_all':True})
deblur_img_n_1 = np.reshape(res_n_1.x, (m, n))
PSNR_n[0]= metrics.peak_signal_noise_ratio(X, deblur_img_n_1)
MSE_n[0] = mean_squared_error(X, deblur_img_n_1)

res_n_2 = minimize(f, x0, method='CG', jac=df, options={'maxiter':max_it_2, 'return_all':True})
deblur_img_n_2 = np.reshape(res_n_2.x, (m, n))
PSNR_n[1]= metrics.peak_signal_noise_ratio(X, deblur_img_n_2)
MSE_n[1]= mean_squared_error(X, deblur_img_n_2)

res_n_3 = minimize(f, x0, method='CG', jac=df, options={'maxiter':max_it_3, 'return_all':True})
deblur_img_n_3 = np.reshape(res_n_3.x, (m, n))
PSNR_n[2]= metrics.peak_signal_noise_ratio(X, deblur_img_n_3)
MSE_n[2]= mean_squared_error(X,deblur_img_n_3)


PSNR_r = np.zeros(3)
MSE_r = np.zeros(3)
lambda_ = 0.01

res_r_1 = minimize(f1, b, lambda_, method= 'CG', jac=df1, options={'maxiter':max_it_1,'return_all':True})
deblur_img_r_1 = np.reshape(res_r_1.x, (m, n))
PSNR_r[0]= metrics.peak_signal_noise_ratio(X, deblur_img_r_1)
MSE_r[0] = mean_squared_error(X, deblur_img_r_1)

res_r_2 = minimize(f1, b, lambda_, method= 'CG', jac=df1, options={'maxiter':max_it_2,'return_all':True})
deblur_img_r_2 = np.reshape(res_r_2.x, (m, n))
PSNR_r[1]= metrics.peak_signal_noise_ratio(X, deblur_img_r_2)
MSE_r[1] = mean_squared_error(X, deblur_img_r_2)

res_r_3 = minimize(f1, b, lambda_, method= 'CG', jac=df1, options={'maxiter':max_it_3,'return_all':True})
deblur_img_r_3 = np.reshape(res_r_3.x, (m, n))
PSNR_r[2]= metrics.peak_signal_noise_ratio(X, deblur_img_r_3)
MSE_r[2] = mean_squared_error(X, deblur_img_r_3)


plt.figure(figsize = (10,15))
plt.plot(ite, PSNR_n, label = 'PSNR naive', color = 'blue')
plt.plot(ite, PSNR_r, label = 'PSNR regolarizzato', color = 'green')
plt.title('Valori PSNR al variare delle iterazioni' )
plt.legend(loc = 'upper right')
plt.show()

plt.figure(figsize = (10,15))
plt.plot(ite, MSE_n, label = 'MSE naive', color = 'blue')
plt.plot(ite, MSE_r, label = 'MSE regolarizzato', color = 'green')
plt.title('Valori  MSE al variare delle iterazioni')
plt.legend(loc = 'upper right')
plt.show()

plt.figure(figsize = (20, 20))

img1 = plt.subplot(2, 2, 1)
img1.imshow(X, cmap = 'gray')
plt.title('Immagine Originale')
imgs = plt.subplot(2, 2, 2)
imgs.imshow(b, cmap = 'gray')
plt.title('Immagine Corrotta')
imgn = plt.subplot(2, 2, 3)
imgn.imshow(deblur_img_n_3, cmap = 'gray')
plt.title('Immagine naive')
imgr = plt.subplot(2, 2, 4)
imgr.imshow(deblur_img_r_3, cmap = 'gray')
plt.title('Immagine Tikhonov')
plt.show()


#discutere i risultati di tikhonov al variare di lambda
plt.figure(figsize = (15, 10))

fig = plt.subplot(2,2,1)
fig.imshow(X, cmap = 'gray')
plt.title('Immagine originale')

res_r = minimize(f1, b, 0.1, method= 'CG', jac=df1, options={'maxiter':max_it_3,'return_all':True})
deblur_img_r = np.reshape(res_r.x, (m, n))
fig1 = plt.subplot(2,2,2)
fig1.imshow(deblur_img_r, cmap = 'gray')
plt.title('Lambda = 0.1')

res_r_1 = minimize(f1, b, 0.01, method= 'CG', jac=df1, options={'maxiter':max_it_3,'return_all':True})
deblur_img_r_1 = np.reshape(res_r_1.x, (m, n))
fig2 = plt.subplot(2,2,3)
fig2.imshow(deblur_img_r_1, cmap = 'gray')
plt.title('Lambda = 0.01')



res_r_2 = minimize(f1, b, 0.001, method= 'CG', jac=df1, options={'maxiter':max_it_3,'return_all':True})
deblur_img_r_2 = np.reshape(res_r_2.x, (m, n))
fig2 = plt.subplot(2,2,4)
fig2.imshow(deblur_img_r_2, cmap = 'gray')
plt.title('Lambda = 0.001')






