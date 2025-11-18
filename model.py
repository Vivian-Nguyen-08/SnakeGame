import numpy as np 
import os 
from classifier import Classifier 
from layers import fc_forward, fc_backward, relu_forward,relu_backward 
from optim import SGD

class Linear_QNet(Classifier): 
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
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
    
    def save(self, file_name='model.pkl'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        super().save(file_path)
    
    @classmethod
    def load(cls,file_name='model.pkl'): 
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)
        model = super(Linear_QNet, cls).load(file_path)
        return model 
    

class QTrainer: 
    def __init__(self,model,lr,gamma): 
        self.lr = lr 
        self.gamma = gamma 
        self.model = model 
        self.optimizer = SGD(model.parameters(),learning_rate=lr)

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)
        reward = np.array(reward)

        if len(state.shape) == 1:
            state = state.reshape(1, -1)
            next_state = next_state.reshape(1, -1)
            action = action.reshape(1, -1)
            reward = reward.reshape(1, -1)
            done = [done]

        # predict Q values with current state 
        pred, cache = self.model.forward(state)

        target = pred.copy()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_q, _ = self.model.forward(next_state[idx:idx+1])
                # bellman equation 
                Q_new = reward[idx] + self.gamma * np.max(next_q)
            
            target[idx][np.argmax(action[idx])] = Q_new
    
        loss, grad_pred = self.mse_loss(pred, target)
        grads = self.model.backward(grad_pred, cache)
        self.optimizer.step(grads)
        
        return loss 
    
    def mse_loss(self,pred,target): 
        loss = np.sum((pred - target) ** 2) / (2 * pred.shape[0])
        grad = (pred - target) / (pred.shape[0])

        return loss,grad 