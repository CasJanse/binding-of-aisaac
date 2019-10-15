import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, name, inputNodes, hiddenNodes, outputNodes, learningRate, hiddenLayers):
        self.name = name

        # Set the number of nodes in each input, hidden, output layer
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        self.hidden_layers = hiddenLayers

        # Learning rate
        self.lr = learningRate

        # Initiate weights list
        self.weights = [0] * (self.hidden_layers + 1)

        # Initiate weights between input layer and first hidden layer
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.weights[0] = self.wih

        # Initiate weights between hidden layers
        for x in range(self.hidden_layers - 1):
            self.weights[(x + 1)] = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.hnodes))

        # Initiate weights between last hidden layer and output layer
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.weights[len(self.weights) - 1] = self.who

        # Initialise activation function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # Convert input list to 2d array
        output_array = [0] * (self.hidden_layers + 2)
        inputs = np.array(inputs_list, ndmin=2).T
        output_array[0] = inputs
        targets = np.array(targets_list, ndmin=2).T

        # Calculate output of every layer
        for x in range(len(output_array) - 1):
            output_array[(x + 1)] = self.activation_function(np.dot(self.weights[x], output_array[x]))

        # hidden_inputs = np.dot(self.wih, inputs)
        # # Calculate hidden layer outputs with activation function
        # hidden_outputs = self.activation_function(hidden_inputs)
        #
        # # Calculate signals coming into output layer
        # final_inputs = np.dot(self.who, hidden_outputs)
        # # Calculate output layer outputs with the activation function
        # final_outputs = self.activation_function(final_inputs)

        # Calculate the error
        error_array = [0] * (len(output_array) - 1)
        error_array[-1] = targets - output_array[-1]

        for x in range(len(error_array) - 1):
            error_array[-(x + 2)] = np.dot(self.weights[-(x + 1)].T, error_array[-(x + 1)])

        # # Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        # hidden_errors = np.dot(self.who.T, output_errors)

        # Update all weights based on their errors
        for x in range(len(self.weights)):
            self.weights[-(x + 1)] += self.lr * np.dot((error_array[-(x + 1)] * output_array[-(x + 1)] * (1.0 - output_array[-(x + 1)])), np.transpose(output_array[-(x + 2)]))

        # # Update the weights for the links between the hidden and output layers
        # self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        #
        # # Update the weights for the links between the input layer and the hidden layer
        # self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs_list):
        # Convert input list to 2d array
        output_array = [0] * (self.hidden_layers + 2)
        inputs = np.array(inputs_list, ndmin=2).T
        output_array[0] = inputs

        # Calculate output of every layer
        for x in range(len(output_array) - 1):
            output_array[(x + 1)] = self.activation_function(np.dot(self.weights[x], output_array[x]))

        return output_array[-1].flatten()

    def save_weights(self):
        for i in range(len(self.weights)):
            np.savetxt('networks/weights_{}_{}.csv'.format(i, self.name), self.weights[i], delimiter=',', fmt='%f')
        pass

    def load_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.genfromtxt('networks/weights_{}_{}.csv'.format(i, self.name), delimiter=',')
        pass
