# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:47:41 2021

@author: felix
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size; 


class model:
    
    def __init__(self):
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime
        
        
    def add(self,layer):
        self.layers.append(layer)
    
    def predict(self,x):
        
        for layer in self.layers:
            x = layer.forward(x)
        return x   
    
    def train(self, x, y, epochs, lr):
        a,_ = x.shape
        b,_ = y.shape
        
        if a != b:
            raise AttributeError("Requires x and y data to be equal in first dimension, that is to have the equal number of samples")
        else:
            for i in range(epochs):
                error_prime = 0
                error = 0
                len_data = len(x)
                for j,data in enumerate(x):
                    output = self.predict(np.atleast_2d(data))
                    error += self.loss(y[j],output)
                    
                    
                    error_prime = self.loss_prime(y[j],output)
                
                    #error_prime /= epochs
                    #print(i,error_prime)
                    for layer in reversed(self.layers):
                        error_prime = layer.backward(lr,error_prime)
                    
                print(error/len_data)
    
    
    
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
        
        
        
class FullyConnected(Layer):
    def __init__(self,inp_size,outp_size):
        self.weights = np.random.rand(inp_size,outp_size)-0.5
        self.bias = np.random.rand(1,outp_size)-0.5
        
    def forward(self,inp):
        self.input = inp
        return (inp @ self.weights)+self.bias
        
    def backward(self,lr,error):
        self.bias = self.bias - lr*error
        self.weights = self.weights - lr*(np.transpose(self.input) @ error)
        return error @ np.transpose(self.weights)
        
        
        
class SigmoidLayer(Layer):
     
   
    def __sgm(self,x):
        return 1/(1+np.exp(x))
    
    def __sgm_prime(self,x):
        return self.__sgm(x) * (1 - self.__sgm(x))
   
    def forward(self,inp):
        self.input_data = inp
        return self.__sgm(inp)
    
    def backward(self,lr,error):
        return self.__sgm_prime(self.input_data)*error
   
class TanHLayer(Layer):
     
   
    def __tanh(self,x):
        return np.tanh(x)
    
    def __tanh_prime(self,x):
        return 1-np.tanh(x)**2
   
    def forward(self,inp):
        self.input_data = inp
        return self.__tanh(inp)
    
    def backward(self,lr,error):
        return self.__tanh_prime(self.input_data)*error



#for n in range(1,1000):
    
   
                   
nn = model()

nn.add(FullyConnected(1,50))        
nn.add(TanHLayer())   
nn.add(FullyConnected(50,1)) 
     



x_train = np.random.rand(100)*4-2

x_train = np.transpose(x_train)

f = lambda x: 8*x**3+17#np.sin(x)
y_train = f(x_train)

y_train = np.transpose(y_train)

#x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
#y_train = np.array([[0], [1], [1], [0]]) 

nn.train(np.transpose(np.atleast_2d(x_train)),np.transpose(np.atleast_2d(y_train)),1000,0.01)


##############

x_test = np.linspace(-2,2,num=1000)


y_pred = nn.predict((np.transpose(np.atleast_2d(x_test))))

plt.figure()

plt.scatter(x_train,y_train)
plt.scatter(x_test,y_pred)


