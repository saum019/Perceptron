
#@author Bhatt, Saumya

import numpy as np

class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):
      
        self.input_dimensions=input_dimensions
        self.number_of_nodes=number_of_nodes
        self.weights = []
        self.initialize_weights()
        
    def initialize_weights(self,seed=None):
        if seed != None:
            np.random.seed(seed)
        self.weights = np.array(self.weights, dtype=np.float)
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions+1)
        #print(self.weights)

    def set_weights(self, W):

        if len(W)==self.number_of_nodes and len(W[0])==self.input_dimensions+1:
            self.weights=W
            return None 
        else:
            return -1
        
    def get_weights(self):

        return self.weights
    
    def predict(self, X):
   
        "y=xw"
        X = np.insert(X,0,1,axis=0)
        multi = np.dot(self.weights,X)
        predicted = np.where(multi[:] <0, 0,1)
        return predicted

    def train(self, X, Y, num_epochs=10,  alpha=0.1):
      
        X = np.insert(X,0,1,axis=0);
        print(X[:,0])
        for i in range(num_epochs):
           for j in range(len(X[0])):
              
               x_sliced = np.expand_dims(X[:,j],axis =1)
               output = np.dot(self.weights,x_sliced)
               predicted_value = np.where(output[:] <0, 0,1) 
               target_value = np.expand_dims(Y[:,j],axis=1)
               #print(target_value.shape,predicted_value.shape)
               error = target_value - predicted_value
               ep = np.dot(error,np.transpose(x_sliced))
               self.weights = self.weights + alpha*(ep)

    def calculate_percent_error(self,X, Y):
  
        predicted= self.predict(X);
        count = 0;
        for i in range(len(X[0])):
           predicted_value = np.expand_dims(predicted[:,i],axis=1);
           target_value = np.expand_dims(Y[:,i],axis=1);
           if(np.array_equal(predicted_value,target_value)):
               count +=0
           else:
               count+=1;
        return (count/len(X[0])*100);

if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())
