# NAME:     Darla Maneja
# EMAIL:    djm160830@utdallas.edu
# SECTION:  CS4372.001
# Assignment 2 nn-enhanced

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px # data viz
import pdb


class NeuralNet:
    def __init__(self, dataFile, header=True, h=4):
        #np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h represents the number of neurons in the hidden layer
        self.scaler = MinMaxScaler()
        raw_input = pd.read_csv(dataFile)
        # TODO: Remember to implement the preprocess method
        processed_data = self.preprocess(raw=raw_input)
        self.train_dataset, self.test_dataset = train_test_split(processed_data, test_size=.20, random_state=0)
        ncols = len(self.train_dataset.columns)
        nrows = len(self.train_dataset.index)
        self.X = self.preprocess(x=self.train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1))
        self.y = y=self.train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)

        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[1])
        if not isinstance(self.y[0], np.ndarray):
            self.output_layer_size = 1
        else:
            self.output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.W_hidden = 2 * np.random.random((input_layer_size, h)) - 1
        self.Wb_hidden = 2 * np.random.random((1, h)) - 1

        self.W_output = 2 * np.random.random((h, self.output_layer_size)) - 1
        self.Wb_output = np.ones((1, self.output_layer_size))

        self.deltaOut = np.zeros((self.output_layer_size, 1))
        self.deltaHidden = np.zeros((h, 1))
        self.h = h


    """
    TODO: preprocess() - Preprocesses the data by dropping outliers, normalization
    """
    def preprocess(self, **kwargs):
        d = kwargs.get('raw', pd.DataFrame())
        x = kwargs.get('x', np.array([]))
        xtest = kwargs.get('xtest', np.array([]))
        y = kwargs.get('y', np.array([]))

        # Drop outliers
        if not d.empty:
            d.drop(index=d.loc[d['creatinine_phosphokinase']>3000,:].index, inplace=True)
            d.drop(index=d.loc[d['platelets']>600000,:].index, inplace=True)
            return d

        # Scale
        if x.size != 0:
            x = self.scaler.fit_transform(x)
            return x
        if xtest.size != 0:
            xtest = self.scaler.transform(xtest)
            return xtest


    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __relu(self, x):
        return np.maximum(x, 0)

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    def __relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x



    # Below is the training function. Added parameters for adam gd optimizer.
    def train(self, activation, max_iterations=60000, learning_rate=0.001, epsilon=10**(-8), beta_1=0.9, beta_2=0.999, \
        m1=0, m2=0, m3=0, m4=0, v1=0, v2=0, v3=0, v4=0):
        print(f'ACTIVATION: {activation}')
        for iteration in range(max_iterations):
            out = self.forward_pass(activation=activation)
            error = 0.5 * np.power((out - self.y), 2)

            # TODO: I have coded the sigmoid activation, you have to do the rest
            self.backward_pass(out, activation=activation)

            # ADAM GRADIENT DESCENT OPTIMIZER
            gradient = np.dot(self.X_hidden.T, self.deltaOut)
            m1 = beta_1*m1 + (1-beta_1)*gradient
            v1 = beta_2*v1 + (1-beta_2)*gradient**2
            m_hat1 = m1/(1-beta_1**(iteration+1))
            v_hat1 = v1/(1-beta_2**(iteration+1))
            update_weight_output = (learning_rate * m_hat1) / (np.sqrt(v_hat1) + epsilon)
            
            gradient = np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaOut)
            m2 = beta_1*m2 + (1-beta_1)*gradient
            v2 = beta_2*v2 + (1-beta_2)*gradient**2
            m_hat2 = m2/(1-beta_1**(iteration+1))
            v_hat2 = v2/(1-beta_2**(iteration+1))
            update_weight_output_b = (learning_rate * m_hat2) / (np.sqrt(v_hat2) + epsilon) 
            
            gradient = np.dot(self.X.T, self.deltaHidden)
            m3 = beta_1*m3 + (1-beta_1)*gradient
            v3 = beta_2*v3 + (1-beta_2)*gradient**2
            m_hat3 = m3/(1-beta_1**(iteration+1))
            v_hat3 = v3/(1-beta_2**(iteration+1))
            update_weight_hidden = (learning_rate * m_hat3) / (np.sqrt(v_hat3) + epsilon) 
            
            gradient = np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden)
            m4 = beta_1*m4 + (1-beta_1)*gradient
            v4 = beta_2*v4 + (1-beta_2)*gradient**2
            m_hat4 = m4/(1-beta_1**(iteration+1))
            v_hat4 = v4/(1-beta_2**(iteration+1))
            update_weight_hidden_b = (learning_rate * m_hat4) / (np.sqrt(v_hat4) + epsilon)


            self.W_output += update_weight_output
            self.Wb_output += update_weight_output_b
            self.W_hidden += update_weight_hidden
            self.Wb_hidden += update_weight_hidden_b


        # print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden))
        # print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_output))

        # print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden))
        # print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_output))
        # np.set_printoptions(precision=3, suppress=True)
        # print(f'\nThe predictions are:\n{out}\n')
        print(f'\nAfter {str(max_iterations)} iterations, total error with {activation} activation function: {str(np.sum(error))}')

    def forward_pass(self, activation):
        # pass our inputs through our neural network
        in_hidden = np.dot(self.X, self.W_hidden) + self.Wb_hidden
        # TODO: I have coded the sigmoid activation, you have to do the rest
        if activation == "sigmoid":
            self.X_hidden = self.__sigmoid(in_hidden)
        elif activation == "tanh":
            self.X_hidden = self.__tanh(in_hidden)
        elif activation == "relu":
            self.X_hidden = self.__relu(in_hidden)            

        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
        if activation == "sigmoid":
            out = self.__sigmoid(in_output)
        elif activation == "tanh":
            out = self.__tanh(in_output)
        elif activation == "relu":
            out= self.__relu(in_output)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__sigmoid_derivative(self.X_hidden))
        elif activation == "tanh":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__tanh_derivative(self.X_hidden))
        elif activation == "relu":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__relu_derivative(self.X_hidden))

        self.deltaHidden = delta_hidden_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, activation, header = True):
        ncols = len(self.test_dataset.columns)
        nrows = len(self.test_dataset.index)
        self.X = self.preprocess(xtest=self.test_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1))
        self.y = self.test_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)

        print(f'ACTIVATION: {activation}')
        out = self.forward_pass(activation=activation)
        error = 0.5 * np.power((out - self.y), 2)
        # TODO: obtain prediction on self.test_dataset
        # np.set_printoptions(precision=3, suppress=True)
        # print(f'\nThe predictions are:\n{out}')

        return error



if __name__ == "__main__":
    # perform pre-processing of both training and test part of the test_dataset
    # split into train and test parts if needed
    neural_network = NeuralNet("https://raw.githubusercontent.com/djm160830/nn-enhanced/master/heart_failure_clinical_records_dataset.csv?token=AJHKVR77C5RZXKFWYJT5UC27P53DC")
    neural_network.train(activation="sigmoid")
    testError = neural_network.predict(activation="sigmoid")
    print("Total Test error with sigmoid activation:                             " + str(np.sum(testError)))

    neural_network = NeuralNet("https://raw.githubusercontent.com/djm160830/nn-enhanced/master/heart_failure_clinical_records_dataset.csv?token=AJHKVR77C5RZXKFWYJT5UC27P53DC")
    neural_network.train(activation="tanh")
    testError = neural_network.predict(activation="tanh")
    # np.set_printoptions(precision=3, suppress=True)
    print("Total Test error with tanh activation:                             " + str(np.sum(testError)))

    neural_network = NeuralNet("https://raw.githubusercontent.com/djm160830/nn-enhanced/master/heart_failure_clinical_records_dataset.csv?token=AJHKVR77C5RZXKFWYJT5UC27P53DC")
    neural_network.train(activation="relu")
    testError = neural_network.predict(activation="relu")
    print("Total Test error with relu activation:                             " + str(np.sum(testError)))