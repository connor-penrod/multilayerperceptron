
import numpy as np
from io import BytesIO
import math, random

class MultiLayerPerceptron(object):
   
    
    def __init__(self, i_count, h_count, o_count, learning_rate, momentum):
        self.i_count = i_count
        self.h_count = h_count
        self.o_count = o_count
        self.samples = None
        self.weights_layer1 = None
        self.weights_layer2 = None
        self.hidden_layer_bias = None
        self.output_layer_bias = None
        self.hidden_layer_aggregation = np.array([])
        self.output_layer_aggregation = np.array([])
        self.hidden_layer_outputs = None
        self.output_layer_outputs = None
        self.targets = None
        self.output_layer_delta = None
        self.hidden_layer_delta = None
        self.output_errors = []
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prev_layer2_change = np.zeros((self.h_count, self.o_count))
        self.prev_layer1_change = np.zeros((self.i_count, self.h_count))
    
    def test(self):
        print(self.weights_layer1)
        print(self.weights_layer2)
        print(self.hidden_layer_bias)
        print(self.output_layer_bias)
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def d_sigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
        
    def read_data(self, samples_f, weights1_f, weights2_f, bias1_f, bias2_f):
        with open(samples_f) as samples:
            self.samples = np.genfromtxt(BytesIO(samples.read().encode()), delimiter=',', usecols=(0,1,2))
        with open(samples_f) as samples: 
            self.targets = np.genfromtxt(BytesIO(samples.read().encode()), delimiter=',', usecols=(3,4))
        with open(weights1_f) as weights1:
            self.weights_layer1 = np.genfromtxt(BytesIO(weights1.read().encode()), delimiter=',')
        with open(weights2_f) as weights2:
            self.weights_layer2 = np.genfromtxt(BytesIO(weights2.read().encode()), delimiter=',')
        with open(bias1_f) as bias1:
            self.hidden_layer_bias = np.genfromtxt(BytesIO(bias1.read().encode()), delimiter=',')
        with open(bias2_f) as bias2:
            self.output_layer_bias = np.genfromtxt(BytesIO(bias2.read().encode()), delimiter=',')
            
    def forward_pass(self, k):
        #compute hidden layer aggregations
        #self.hidden_layer_aggregation = np.zeros((1,self.h_count))
        self.output_layer_aggregation = np.zeros((1,self.o_count))
        #self.hidden_layer_outputs = np.zeros((1,self.h_count))
        self.output_layer_outputs = np.zeros((1,self.o_count))
        
        self.hidden_layer_aggregation = np.dot(self.samples[k], np.transpose(self.weights_layer1))

        #for weight in range(len(self.weights_layer1)):
         #   self.hidden_layer_aggregation = np.append(self.hidden_layer_aggregation,
          #                                        np.dot(self.samples[k], np.transpose(self.weights_layer1[weight])))
        self.hidden_layer_aggregation += self.hidden_layer_bias
        #compute hidden layer outputs
        self.hidden_layer_outputs = self.sigmoid(self.hidden_layer_aggregation)
        
        #compute output layer aggregations
        #for weight in range(len(self.weights_layer2)):
         #   self.output_layer_aggregation = np.append(self.output_layer_aggregation,
          #                                        np.dot(self.hidden_layer_outputs, np.transpose(self.weights_layer2[weight])))
        self.output_layer_aggregation = np.dot(self.hidden_layer_outputs, np.transpose(self.weights_layer2))
        self.output_layer_aggregation += self.output_layer_bias
       
        #compute output layer outputs
        self.output_layer_outputs = self.sigmoid(self.output_layer_aggregation)
    
    def calculate_layer2_error(self, k):
        #calculates individual neuron error functions at the output layer
        #calculates delta for output layer
        error = (self.targets[k] - self.output_layer_outputs)
        self.output_layer_delta = error * self.d_sigmoid(self.output_layer_aggregation)
        self.output_layer_bias -= self.output_layer_delta
        for er in error:
            self.output_errors.append(er)
    
    def calculate_layer1_error(self, k):
        #calculates individual neuron error functions at the hidden layer
        #calculates delta for hidden layer
        #error = np.zeros((self.h_count))
        #for neuron in range(len(self.output_layer_delta)):
            #error += (self.output_layer_delta[neuron] * self.weights_layer2[neuron])
        #error = self.output_layer_delta * self.weights_layer2;
        error = np.dot(self.output_layer_delta, self.weights_layer2)
        
        self.hidden_layer_delta = error * self.d_sigmoid(self.hidden_layer_aggregation)
        self.hidden_layer_bias -= self.hidden_layer_delta
    
    def adjust_weights(self, k):
        #apply deltas to weights to update them
        layer1_change = np.array([])
        for w in range(len(self.samples[k])):
            layer1_change = np.append(layer1_change, self.hidden_layer_delta * self.samples[k][w])
        layer1_change = layer1_change.reshape(self.i_count, self.h_count)

        self.weights_layer1 -= np.transpose(layer1_change)*self.learning_rate
        self.weights_layer1 -= self.momentum*np.transpose(self.prev_layer1_change)
            
        layer2_change = np.array([])
        for w in range(len(self.hidden_layer_aggregation)):
            layer2_change = np.append(layer2_change, self.output_layer_delta * self.hidden_layer_outputs[w])
        layer2_change = layer2_change.reshape(self.h_count, self.o_count)
        
        self.weights_layer2 -= np.transpose(layer2_change)*self.learning_rate
        self.weights_layer2 -= self.momentum*np.transpose(self.prev_layer2_change)
        
        self.prev_layer2_change = layer2_change
        self.prev_layer1_change = layer1_change
    
    def calculateTotalError(self, k):
        total = 0
        for er in self.output_errors:
            total += er*er
        total *= 0.5
        return total
    
    def print_sample_error(self, k):
        total_error = 0.5 * self.output_errors[2*k]*self.output_errors[2*k] + self.output_errors[2*k+1]*self.output_errors[2*k+1]
        print(total_error)
        
    def print_variables(self):
        print("Weights 1")
        print(self.weights_layer1)
        print("Weights 2")
        print(self.weights_layer2)   
        print("Bias - Output")
        print(self.output_layer_bias)
        print("Bias - Hidden")
        print(self.hidden_layer_bias)
        print("Output deltas")
        print(self.output_layer_delta)
        print("Hidden deltas")
        print(self.hidden_layer_delta)
        print("Output errors")
        print(self.output_errors[-2:])
        
    def runEpoch(self):
        for k in range(len(self.samples)):
            self.forward_pass(k)
            self.calculate_layer2_error(k)
            self.calculate_layer1_error(k)
            self.adjust_weights(k)
            self.print_sample_error(k)
            #self.print_variables(k)
        totalError = n.calculateTotalError(k)
        print(totalError)
    def scrambleSamples(self):
        self.samples = random.shuffle(self.samples)
    
    def runEpochsIndefinitely(self):
        
        while(1):
            self.runEpoch()
            self.scrambleSamples()          

n = MultiLayerPerceptron(3,11,2,0.7,0.3)
n.read_data("cross_data (3 inputs - 2 outputs).csv",
            "w1 (3 inputs - 11 nodes).csv",
            "w2 (from 11 to 2).csv",
            "b1 (11 nodes).csv",
            "b2 (2 output nodes).csv")

n.runEpoch()

