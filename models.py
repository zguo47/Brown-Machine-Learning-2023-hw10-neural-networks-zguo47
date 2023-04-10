import random
import numpy as np


def l2_loss(predictions, Y):
    '''
    Computes L2 loss (sum squared loss) between true values, Y, and predictions.
    :param predictions: A 1D Numpy array of the same size of Y
    :param Y: 1D Numpy array
    :return: L2 loss using predictions for Y.
    '''
    # TODO: implement the squared error loss function
    return np.sum((predictions - Y)**2)


def sigmoid(a):
    '''
    Sigmoid function, given by sigma(a) =  1/(1 + exp(-a))
    :param a: a scalar or Numpy array
    :return: sigmoid function evaluated at a (applied element-wise if it is an array)
    '''
    return np.where(a > 0, 1 / (1 + np.exp(-a)), np.exp(a) / (np.exp(a) + np.exp(0)))


def sigmoid_derivative(a):
    '''
    First derivative of the sigmoid function with respect to a.
    :param a: a scalar or Numpy array
    :return: derivative of sigmoid evaluated at a (applied element-wise if it is an array)
    '''
    # TODO: implement the first derivative of the sigmoid function
    return sigmoid(a)*(1-sigmoid(a))


class OneLayerNN:
    '''
    One layer neural network trained with Stocastic Gradient Descent (SGD)
    '''

    def __init__(self):
        '''
        @attrs:
            w: The weights of the first layer of the neural network model
            learning_rate: The learning rate to use for SGD
            epochs: The number of times to pass through the dataset
            o: The output of the network
        '''
        # initialize self.w in train()
        self.w = None
        self.learning_rate = 0.001
        self.epochs = 25
        self.o = None

    def train(self, X, Y, print_loss=True):
        '''
        Training loop with SGD for OneLayerNN model.
        :param X: 2D Numpy array, each row representing one example
        :param Y: 1D Numpy array, each entry a label for the corresponding row (example) in X
        :param print_loss: If True, print the loss after each epoch.
        :return: None
        '''
        # TODO: initialize self.w. In the OneLayerNN, you can assume that every example
        # has a 1 as the last feature, so no separate bias term is needed.

        # TODO: Train network for certain number of epochs defined in self.epochs
        # TODO: Use a variable called epoch to denote the current epoch's number

        # TODO: Shuffle the examples (X) and labels (Y)

        # TODO: We need to iterate over each example for each epoch
        # (This is equivalent to using a batch size of one.)

        # TODO: Perform the forward and backward pass on the current example
        self.w = np.random.uniform(0, 1, X.shape[1])
        epoch = 0
        for e in range(self.epochs):
            epoch = e
            p = np.random.permutation(len(X))
            X = X[p]
            Y = Y[p]
            for i in range(len(Y)):
                X_b = X[i]
                Y_b = Y[i]
                self.forward_pass(X_b)
                self.backward_pass(X_b, Y_b)

        # Print the loss after every epoch
            if print_loss:
                print('Epoch: {} | Loss: {}'.format(epoch, self.loss(X, Y)))

    def forward_pass(self, x):
        '''
        Computes the output of the network given an input vector x and stores the result in self.o.
        :param x: 1D Numpy array, representing one example
        :return: None
        '''
        self.o = np.dot(x, self.w)

    def backward_pass(self, x, Y):
        '''
        First computes the gradient of the loss on an example with respect to self.w.
        Then updates self.w. Should only be called after self.forward_pass. 
        :param x: 1D Numpy array, representing one example
        :param y: scalar, the label for the example x
        :return: None
        '''
        # TODO: Calculate the gradient of the weights

        # TODO: Update the weights using the gradient
        dl_dz = (sigmoid(np.dot(x, self.w))-Y)*sigmoid_derivative(np.dot(x, self.w))
        d_loss = np.dot(np.transpose(x), dl_dz)
        self.w = self.w - self.learning_rate*d_loss

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).
        :param X: 2D Numpy array, each row representing one example
        :param Y: 1D Numpy array, each entry a label for the corresponding row (example) in X
        :return: A float which is the squared error of the model on the dataset
        '''
        # Perform the forward pass and compute the l2 loss
        outputs = np.zeros([X.shape[0]])
        for i in range(X.shape[0]):
            self.forward_pass(X[i])
            outputs[i] = self.o

        return l2_loss(outputs, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).
        MSE = Total squared error / # of examples
        :param X: 2D Numpy array, each row representing one example
        :param Y: 1D Numpy array, each entry a label for the corresponding row (example) in X
        :return: A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y) / X.shape[0]

class TwoLayerNN:
    '''
    Two layer neural network trained with Stochastic Gradient Descent (SGD)
    '''

    def __init__(self, hidden_size, activation=sigmoid, activation_derivative=sigmoid_derivative):
        '''
        @attrs:
            activation: the activation function applied after the first layer
            activation_derivative: the derivative of the activation function. Used for training.
            hidden_size: The hidden size of the network (an integer)
            learning_rate: The learning rate to use for SGD
            epochs: The number of times to pass through the dataset
            
            Other variable naming conventions: 
                Letters: 
                    w = weights layers
                    b = bias layers
                    a = output of the first layer computed during forward pass
                    o = the activated output of the first layer computed during the forward pass (activation of a)
                Numbers: 
                    represent the number layer 
                        Example: W01 is the weights between layers 0 and 1
        '''
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size
        self.learning_rate = 0.01
        self.epochs = 25

        # initialize the following weights and biases in the train() method
        self.W01 = None
        self.b1 = None
        self.W12 = None
        self.b2 = None

        # Outputs of each layer
        self.a1 = None
        self.o1 = None
        self.a2 = None
        self.o2 = None

    def train(self, X, Y, print_loss=True):
        '''
        Trains the TwoLayerNN with SGD using Backpropagation.
        :param X: 2D Numpy array, each row representing one example
        :param Y: 1D Numpy array, each entry a label for the corresponding row (example) in X
        :param print_loss: If True, print the loss after each epoch.
        :return: None
        '''
        # NOTE:
        # Use numpy arrays of the following dimensions for your model's parameters.
        # layer 1 weights (W01): hidden_size x input_size (2D matrix)
        # layer 1 bias (b1): hidden_size (1D vector)
        # layer 2 weights (W12): 1 x hidden_size (2D matrix)
        # layer 2 bias (b2): 1 (1D vector)
        # HINT: for best performance initialize weights with np.random.normal or np.random.uniform

        # TODO: Weight and bias initialization

        # TODO: Train network for certain number of epochs
        # TODO: Use a variable called epoch to denote the current epoch's number
        
        # TODO: Shuffle the examples (X) and labels (Y)

        # TODO: We need to iterate over each example for each epoch
        # (This is equivalent to using a batch size of one.)

        # TODO: Perform the forward and backward pass on the current example

        # Print the loss after every epoch
        if print_loss:
            print('Epoch: {} | Loss: {}'.format(epoch, self.loss(X, Y)))


    def forward_pass(self, x):
        '''
        Computes the outputs of the network given an input vector x. Stores the activation function input
        (weighted sum for each neuron) for each layer in self.a* and the post-activation output for each
        layer in self.o*.
        :param x: 1D Numpy array, representing one example
        :return: None
        '''
        # TODO: Calculate output of neural network on X
        pass

        
    def backprop(self, x, y):
        '''
        Calculates the gradients of the loss w.r.t the weights and biases of each layer
        using the backpropagation algorithm
        :param x: 1D Numpy array, representing one example
        :param y: scalar, the label for the example x
        :return: tuple: (W12_grad, b2_grad, W01_grad, b1_grad)
        '''
        # TODO: Implement backpropagation
        # Hint: Use the values computed and saved in self.forward_pass
        pass

    def backward_pass(self, x, y):
        '''
        First computes the gradient of the loss on an example with respect to self.W01, self.b1, self.W12,
        and self.b2 by calling self.backprop. Then updates all those parameters. Should only be called
        after self.forward_pass.
        :param x: 1D Numpy array, representing one example
        :param y: scalar, the label for the example x
        :return: None
        '''
        # TODO: Compute the gradients for the model's weights by calling self.backprop

        # TODO: Update the weights using gradient descent
        pass


    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).
        :param X: 2D Numpy array, each row representing one example
        :param Y: 1D Numpy array, each entry a label for the corresponding row (example) in X
        :return: A float which is the squared error of the model on the dataset
        '''
        # Perform the forward pass and compute the l2 loss
        outputs = np.zeros([X.shape[0]])
        for i in range(X.shape[0]):
            self.forward_pass(X[i])
            outputs[i] = self.o2

        return l2_loss(outputs, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).
        MSE = (Total squared error) / (# of examples)
        :param X: 2D Numpy array, each row representing one example
        :param Y: 1D Numpy array, each entry a label for the corresponding row (example) in X
        :return: A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y) / X.shape[0]
