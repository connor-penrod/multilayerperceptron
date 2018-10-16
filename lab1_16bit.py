
import math
import csv
import random
from pprint import pprint
import sys
import numpy as np

input_node_count = 3
hidden_node_count = 11
output_node_count = 2

samples = []
w_layer1 = []
w_layer2 = []
dw_layer1 = [[0] * input_node_count] * (hidden_node_count+1)
dw_layer2 = [[0] * hidden_node_count] * (output_node_count+1)
hidden_layer_bias = []
output_layer_bias = []
targets = []
learning_rate = 0.7
momentum = 0.3
er = []

#test targets for debugging
test_targets = [[0,1]] * 400

def activation_function(x):
    return 1/(1 + math.exp(-x))
def deriv_activation_function(x):
    return activation_function(x)*(1-activation_function(x))
    
    
#handles file input
with open('cross_data (3 inputs - 2 outputs).csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        entry = []
        targetSet = []
        for i in range(len(row)):
            if i < len(row) - 2:
                entry.append(np.float16(row[i]))
            else:
                targetSet.append(np.float16(row[i]))
        samples.append(entry)
        targets.append(targetSet)
with open('w1 (3 inputs - 11 nodes).csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        entry = []
        for i in range(len(row)):
            entry.append(np.float16(row[i]))
        w_layer1.append(entry)
with open("w2 (from 11 to 2).csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        entry = []
        for i in range(len(row)):
            entry.append(np.float16(row[i]))
        w_layer2.append(entry)
with open("b1 (11 nodes).csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        entry = np.float16(row[0])
        hidden_layer_bias.append(entry)
with open("b2 (2 output nodes).csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        entry = np.float16(row[0])
        output_layer_bias.append(entry)

#begin multilayer perceptron
x = 0
while(x < 2000): #run through 2000 epochs (disabled for this assignment)
# for each epoch, loop through all samples
    for k in range(len(samples)):
        hidden_layer_aggregation = []
        output_layer_aggregation = []
        hidden_layer_outputs = []
        output_layer_outputs = []
        output_layer_errors = []
        
        # compute aggregate values for input -> hidden layer
        for currentHiddenNode in range(hidden_node_count):
            aggregate = np.float16(0)
            for weight in range(len(w_layer1[currentHiddenNode])):
                aggregate += np.float16(samples[k][weight] * w_layer1[currentHiddenNode][weight])
            aggregate += hidden_layer_bias[currentHiddenNode]
            hidden_layer_aggregation.append(aggregate)
            
        # find hidden layer outputs
        for aggregateIdx in range(hidden_node_count):
            output = np.float16(activation_function(hidden_layer_aggregation[aggregateIdx]))
            hidden_layer_outputs.append(output)
        
        # compute aggregate values for hidden -> output layer
        for currentOutputNode in range(output_node_count):
            aggregate = np.float16(0)
            for weight in range(len(w_layer2[currentOutputNode])):
                aggregate += np.float16(hidden_layer_outputs[weight] * w_layer2[currentOutputNode][weight])
            aggregate += output_layer_bias[currentOutputNode]
            output_layer_aggregation.append(aggregate)
            
        # find output layer outputs
        for aggregateIdx in range(output_node_count):
            output = np.float16(activation_function(output_layer_aggregation[aggregateIdx]))
            output_layer_outputs.append(output)
        
        
        # forward pass done, begin backpropagation   
        
        # start at output neurons, compute weight changes and new weights for weight layer 2
        for output_neuron in range(output_node_count):
            dw_entry = [] #stores the previous weight change
            e_output = (targets[k][output_neuron] - output_layer_outputs[output_neuron]) #calculates output node error
            output_layer_errors.append(e_output) #stores error for hidden layer weight change calculation
            er.append(e_output)
            output_activation_deriv = np.float16(deriv_activation_function(output_layer_aggregation[output_neuron])) #calculates the output layer aggregates 
                                                                                                         #put through the derivative of the sigmoid
            for weight in range(len(w_layer2[output_neuron])): #for each weight in the layer 2
                dw = -learning_rate * -e_output * output_activation_deriv * hidden_layer_outputs[weight] #calculate change in weight
                w_layer2[output_neuron][weight] += dw + momentum * dw_layer2[output_neuron][weight] #update weight with dw and momentum term
                dw_entry.append(dw) #store previous weight change
            db = -learning_rate * -e_output * output_activation_deriv * 1 #find change in bias
            output_layer_bias[output_neuron] += db + momentum*dw_layer2[output_neuron][-1] #update bias
            dw_entry.append(db) #store bias at the end of dw_layer2 array
            dw_layer2[output_neuron] = dw_entry #store the change in weights for use in the next sample
            
        #compute weight changes for hidden layer
        hidden_layer_errors = []
        for hidden_neuron in range(hidden_node_count):
            dw_entry = []
            hidden_activation_deriv = np.float16(deriv_activation_function(hidden_layer_aggregation[hidden_neuron])) #calculates the hidden layer aggregates
            sum_of_error_propagation = 0 #initialize sum for error propagation
            for l in range(output_node_count): #for each output node
                e_output = output_layer_errors[l] #grab the error for that node
                deriv_activation_func_output = np.float16(deriv_activation_function(output_layer_aggregation[l]))

                for weight in range(len(w_layer2[0])): #for each weight in layer 2
                    sum_of_error_propagation += w_layer2[l][weight] * -e_output * deriv_activation_func_output #add the summation formula to our sum_of_error_propagation variable
                
            for weight in range(len(w_layer1[hidden_neuron])): #for each weight in layer 1
                dw = -learning_rate * sum_of_error_propagation * samples[k][weight] * hidden_activation_deriv #find change in weight
                w_layer1[hidden_neuron][weight] += dw + momentum*dw_layer1[hidden_neuron][weight] #update that weight
                dw_entry.append(dw)
            db = -learning_rate * sum_of_error_propagation * 1 * hidden_activation_deriv #find change in bias
            hidden_layer_bias[hidden_neuron] += db + momentum*dw_layer1[hidden_neuron][-1] #update bias (bias is tacked onto the end of the dw_layer1 array)
            dw_entry.append(db) #change in bias is held in the dw_layer1 array 
            dw_layer1[hidden_neuron] = dw_entry #save the change in weight for the momentum of next sample
 
        #find sum of sqr errors
        sum_sqr_errors = 0
        for error in output_layer_errors:
            sum_sqr_errors += (error*error)
        #sum_sqr_errors *= 0.5
        
        
        #debugging stuff
        selection = [_ for _ in range(30)] + [_ for _ in range(300, 314)]
        
        if k in selection:
            print("==Sample #" + str(k) + "==")
            print("Sum of sqr errors for sample: {:0.8f}".format(sum_sqr_errors))
            print(str(targets[k][0]) + " -> {:0.8f}".format(output_layer_outputs[0]))
            print(str(targets[k][1]) + " -> {:0.8f}".format(output_layer_outputs[1]))
            print("")
        
    s = 0 #calculate and print sum of sqr errors for total epoch
    for e in er:
        s += e*e
    print(s)
    print("{:0.8f} {:0.8f} {:0.8f} {:0.8f}".format(w_layer1[0][0], w_layer1[0][1], w_layer1[0][2], hidden_layer_bias[0]))

    break; #remove this if doing more than 1 epoch
    x += 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

