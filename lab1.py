
import math
import csv
import random
from pprint import pprint
import sys
import copy
import matplotlib.pyplot as plt

#test targets for debugging
test_targets = [[0,1]] * 400

def activation_function(x):
    return 1/(1 + math.exp(-x))
def deriv_activation_function(x):
    return activation_function(x)*(1-activation_function(x))

#begin multilayer perceptron
class ThreeLayerPerceptron(object):

    def __init__(self, _input_count, _hidden_count, _output_count, _samples, 
                    _targets, _init_weights1, _init_weights2, _init_bias1, _init_bias2,
                    _learning_rate, _momentum):
        self.input_node_count = _input_count
        self.hidden_node_count = _hidden_count
        self.output_node_count = _output_count

        self.samples = _samples
        self.w_layer1 = _init_weights1
        self.w_layer2 = _init_weights2
        self.dw_layer1 = [[0] * self.input_node_count] * (self.hidden_node_count+1)
        self.dw_layer2 = [[0] * self.hidden_node_count] * (self.output_node_count+1)
        self.hidden_layer_bias = _init_bias1
        self.output_layer_bias = _init_bias2
        self.targets = _targets
        self.learning_rate = _learning_rate
        self.momentum = _momentum
        self.errors = []
        self.target_error = 0.001
        self.sumofsqrerrors = []
        
    def train(self):
        prevSumSqr = 0
        currSumSqr = float('infinity')
        epochCount = 1
        while abs(prevSumSqr-currSumSqr) >= self.target_error: #run through until sum sqr errors difference is low enough
        # for each epoch, loop through all self.samples
            
            print("Training epoch #" + str(epochCount))
            epochCount += 1
            
            prevSumSqr = currSumSqr
            
            self.errors = []
            for k in range(len(self.samples)):
                hidden_layer_aggregation = []
                output_layer_aggregation = []
                hidden_layer_outputs = []
                output_layer_outputs = []
                output_layer_errors = []
                
                # compute aggregate values for input -> hidden layer
                for currentHiddenNode in range(self.hidden_node_count):
                    aggregate = 0
                    for weight in range(len(self.w_layer1[currentHiddenNode])):
                        aggregate += (self.samples[k][weight] * self.w_layer1[currentHiddenNode][weight])
                    aggregate += self.hidden_layer_bias[currentHiddenNode]
                    hidden_layer_aggregation.append(aggregate)
                    
                # find hidden layer outputs
                for aggregateIdx in range(self.hidden_node_count):
                    output = activation_function(hidden_layer_aggregation[aggregateIdx])
                    hidden_layer_outputs.append(output)
                
                # compute aggregate values for hidden -> output layer
                for currentOutputNode in range(self.output_node_count):
                    aggregate = 0
                    for weight in range(len(self.w_layer2[currentOutputNode])):
                        aggregate += (hidden_layer_outputs[weight] * self.w_layer2[currentOutputNode][weight])
                    aggregate += self.output_layer_bias[currentOutputNode]
                    output_layer_aggregation.append(aggregate)
                    
                # find output layer outputs
                for aggregateIdx in range(self.output_node_count):
                    output = activation_function(output_layer_aggregation[aggregateIdx])
                    output_layer_outputs.append(output)
                        
                # forward pass done, begin backpropagation   
                curr_w_layer2 = copy.deepcopy(self.w_layer2) #copies current weights for use in hidden layer backpropagation
                # start at output neurons, compute weight changes and new weights for weight layer 2
                for output_neuron in range(self.output_node_count):
                    dw_entry = [] #stores the previous weight change
                    e_output = (self.targets[k][output_neuron] - output_layer_outputs[output_neuron]) #calculates output node error
                    output_layer_errors.append(e_output) #stores error for hidden layer weight change calculation
                    self.errors.append(e_output)
                    output_activation_deriv = deriv_activation_function(output_layer_aggregation[output_neuron]) #calculates the output layer aggregates 
                                                                                                                 #put through the derivative of the sigmoid
                    for weight in range(len(self.w_layer2[output_neuron])): #for each weight in the layer 2
                        dw = -self.learning_rate * -e_output * output_activation_deriv * hidden_layer_outputs[weight] #calculate change in weight
                        self.w_layer2[output_neuron][weight] += dw + self.momentum * self.dw_layer2[output_neuron][weight] #update weight with dw and self.momentum term
                        dw_entry.append(dw + self.momentum * self.dw_layer2[output_neuron][weight]) #+ self.momentum * self.dw_layer2[output_neuron][weight]) #store previous weight change
                    db = -self.learning_rate * -e_output * output_activation_deriv * 1 #find change in bias
                    self.output_layer_bias[output_neuron] += db + self.momentum*self.dw_layer2[output_neuron][-1] #update bias
                    dw_entry.append(db + self.momentum*self.dw_layer2[output_neuron][-1]) #store bias at the end of self.dw_layer2 array
                    self.dw_layer2[output_neuron] = dw_entry #store the change in weights for use in the next sample
                #compute weight changes for hidden layer
                hidden_layer_errors = []
                for hidden_neuron in range(self.hidden_node_count):
                    dw_entry = []
                    hidden_activation_deriv = deriv_activation_function(hidden_layer_aggregation[hidden_neuron]) #calculates the hidden layer aggregates
                    sum_of_error_propagation = 0 #initialize sum for error propagation
                    for output_neuron in range(self.output_node_count): #for each output node
                        e_output = output_layer_errors[output_neuron] #grab the error for that node
                        deriv_activation_func_output = deriv_activation_function(output_layer_aggregation[output_neuron])
                        
                        #this missing bit from Part A right here was the reason for my trouble in Part A. Fixed now.
                        
                        sum_of_error_propagation += curr_w_layer2[output_neuron][hidden_neuron] * deriv_activation_func_output * e_output
                    for weight in range(len(self.w_layer1[hidden_neuron])): #for each weight in layer 1
                        dw = -self.learning_rate * -sum_of_error_propagation * self.samples[k][weight] * hidden_activation_deriv #find change in weight
                        self.w_layer1[hidden_neuron][weight] += dw + self.momentum * self.dw_layer1[hidden_neuron][weight] #update that weight
                        dw_entry.append(dw + self.momentum*self.dw_layer1[hidden_neuron][weight])
                    db = -self.learning_rate * -sum_of_error_propagation * 1 * hidden_activation_deriv #find change in bias
                    self.hidden_layer_bias[hidden_neuron] += db + self.momentum*self.dw_layer1[hidden_neuron][-1] #update bias (bias is tacked onto the end of the self.dw_layer1 array)
                    dw_entry.append(db + self.momentum*self.dw_layer1[hidden_neuron][-1]) #change in bias is held in the self.dw_layer1 array 
                    self.dw_layer1[hidden_neuron] = dw_entry #save the change in weight for the self.momentum of next sample
                                
            s = 0 #calculate and print sum of sqr errors for total epoch
            for e in self.errors:
                s += e*e
            currSumSqr = s
            self.sumofsqrerrors.append(s)
            random.shuffle(self.samples) # randomize data for next epoch
        print("Reached convergence with target error difference = " + str(self.target_error))
    
    def plotSumOfSqrErrors(self):
        plt.plot(self.sumofsqrerrors); plt.xlabel("Epoch"); plt.ylabel("Sum of Sqr. Errors"); plt.title("Sum of Sqr. Errors During Training")
        plt.show()

samples = []
targets = []
w_layer1 = []
w_layer2 = []
hidden_layer_bias = []
output_layer_bias = []
    
#handles file input
with open('cross_data (3 inputs - 2 outputs).csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        entry = []
        targetSet = []
        for i in range(len(row)):
            if i < len(row) - 2:
                entry.append(float(row[i]))
            else:
                targetSet.append(float(row[i]))
        samples.append(entry)
        targets.append(targetSet)
with open('w1 (3 inputs - 11 nodes).csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        entry = []
        for i in range(len(row)):
            entry.append(float(row[i]))
        w_layer1.append(entry)
with open("w2 (from 11 to 2).csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        entry = []
        for i in range(len(row)):
            entry.append(float(row[i]))
        w_layer2.append(entry)
with open("b1 (11 nodes).csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        entry = float(row[0])
        hidden_layer_bias.append(entry)
with open("b2 (2 output nodes).csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        entry = float(row[0])
        output_layer_bias.append(entry)
    
perceptron = ThreeLayerPerceptron(3, 11, 2, samples, targets, w_layer1, w_layer2,
                                hidden_layer_bias, output_layer_bias, 0.7, 0.3)
    
perceptron.train() 
perceptron.plotSumOfSqrErrors()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

