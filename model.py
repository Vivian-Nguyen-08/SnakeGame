import numpy as np 
import os 
from classifier import Classifier 
from layers import fc_forward, fc_backward, relu_forward,relu_backward 
from optim import SGD

class Linear_QNet(Classifier): 
    def __init__(self, input_size,hidden_size,output_size):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """

        weight_scale=1e-3
        #initialized from a gaussian distribution with standard deviation equal to weight_scale 
        self.W1 = np.random.randn(input_size,hidden_size) * weight_scale 
        self.W2 = np.random.randn(hidden_size,output_size) * weight_scale 

        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)
      

    
    def parameters(self):

        # build a dict of all learnable parameters of this model. 
        params = {"W1": self.W1, 
                  "b1": self.b1, 
                  "W2": self.W2, 
                  "b2": self.b2}
        return params

    def forward(self, X):
        out1, cachefc1 = fc_forward(X,self.W1,self.b1)
        out2, cacheRelu  = relu_forward(out1)
        q_values, cachefc2 = fc_forward(out2,self.W2,self.b2)
        cache = (cachefc1,cacheRelu,cachefc2)
        return q_values, cache

    def backward(self, grad_scores, cache):
        cachefc1, cacheRelu,cachefc2 = cache 
        dxfc2, dwfc2, dbfc2 = fc_backward(grad_scores, cachefc2)
        dxRelu = relu_backward(dxfc2,cacheRelu)
        dxfc1, dwfc1, dbfc1 = fc_backward(dxRelu,cachefc1)
        grads = {"W1": dwfc1, 
                  "b1": dbfc1, 
                  "W2": dwfc2, 
                  "b2": dbfc2}
        return grads

    def predict(self,state): 
        q_values,_ = self.forward(state)
        return q_values
    
    