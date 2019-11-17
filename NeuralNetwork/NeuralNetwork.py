import numpy as np
from sklearn.model_selection import train_test_split

from NeuralNetwork.ActivationFunctions import activationFunctionMap
from NeuralNetwork.CostFunctions import CostFunctions, costFunctionMap
from NeuralNetwork.Exceptions import NeuralNetworkException




class NeuralNetwork:
    def __init__(self, layers, activationFunction, costFunction, weightsRange = [-0.01,0.01]):
        """
            Inits a neural network
        :param layers: list of numbers representing the number of neurons on each layer
        :param activationFunction: string or list of activation functions
        :param costFunction: cost function
        :param weightsRange: range of the uniform variable that inits the weights
        """
        #parameter validation
        if len(layers) < 2:
            raise NeuralNetworkException("Number of layers must be above or equal to 2.")
        if len(weightsRange) != 2:
            raise NeuralNetworkException("Parameter 'weightsRange' must be a tuple like (low, high) describing the ranges for the uniform distribution.")

        #subcomponent classes
        self._activationFunctionMap = activationFunctionMap
        self._costFunctonMap = costFunctionMap

        #neural network attributes
        self._layerSizes = layers
        self._weightsRange = weightsRange
        self._layerCount = len(self._layerSizes)  # number of layers in the network
        self._weights = self._initWeights(weightsRange)                                             #list of matrices with weights between each 2 consecutive layers
        self._biases = self._initBiases(weightsRange)
        self._inputShape = (self._layerSizes[0],1)
        self._outputShape = (self._layerSizes[-1],1)
        self._activationFunction = self._initActivationFunctions(activationFunction)    #list of activation function for each layer except first
        self._costFunction = self._initCostFunction(costFunction)                       #cost function for evaluating the network



        #learning attributes
        self._z = [np.ones((self._layerSizes[i],1), dtype=float) for i in range(self._layerCount)]   #neuron values before activation for each layer
        self._a = [np.ones((self._layerSizes[i],1), dtype=float) for i in range(self._layerCount)]   #neuron values after activation for each layer
        self._delta = [np.zeros(weights.shape, dtype=float) for weights in self._weights]          #sums up the errors for each neuron connection for each layer
        self._deltaBias = [np.zeros((self._layerSizes[i],1), dtype=float) for i in range(1, self._layerCount)]

    def _feedforward(self, x, dropoutProb = 0):
        """
            Passes the information through the layers, returning the output.
        :param x: Input data. Must be in column shape, matching the input shape.
        :return: Column matrix representing the output of the last layer.
        """
        x = np.asarray(x)   #converting the input to numpy array
        if x.shape[0] != self._layerSizes[0]:
            if x.shape[1] == self._layerSizes[0]:
                x = x.T
            else:
                raise NeuralNetworkException("Size {} not able to feedforward.".format(x.shape))
        self._z[0] = x
        self._a[0] = x

        for i in range(self._layerCount-1):
            #get the information left in layer i, pass it to layer i+1 through the weight matrix, add the bias
            #then activate with the activation function in layer i+1
            x = np.dot(self._weights[i],x)
            x = x + self._biases[i]
            self._z[i + 1] = x
            x = self._activationFunction[i].activate(x)

            #simulate dropout by zero-ing some of the outputs probabilistically
            if dropoutProb > 0 and i < self._layerCount-2:
                drop = np.random.rand(x.shape[0],x.shape[1])  # Step 1: initialize matrix D1 = np.random.rand(..., ...)
                drop = drop > dropoutProb
                x = np.multiply(x,drop)
                x = x / dropoutProb

            self._a[i+1] = x

        return x


    def _backprop(self, err):
        """
            Runs the backpropagation algorithm, computing the derivative of the cost function with respect to each weight.
            Sums up the errors in variable self._delta
        :param err: The outside error in the output layer.
        :return: None
        """

        err = np.asarray(err)
        #for each layer except input layer, err is the outside error
        #   1. pass the outside error 'inside' the cell, by multiplying the outside error with the
        #           derivative of the activation function with respect to weighted output of the current neuron
        #   2. pass the 'inside' gradient to the outside part of the previous layer, by multiplying
        #           the 'inside' gradient with the weights between the two layers
        #   3. before moving to the previous neuron , compute the derivative of the 'inside' error with respect
        #           to each weight, and sum it to self._delta array

        for i in range(self._layerCount-1, 0, -1):
            #obtain the inside error
            err = err * self._activationFunction[i-1].deriv(self._z[i])

            #compute the gradients for biases and weights
            crtDeltaBiases = err
            crtDeltaWeights = np.dot(err,self._a[i-1].T)

            #sum the delta to self._delta
            self._deltaBias[i-1] = self._deltaBias[i-1] + crtDeltaBiases
            self._delta[i-1] = self._delta[i-1] + crtDeltaWeights

            #continue with previous layer
            err = np.dot(self._weights[i - 1].T,err)


    def _batchUpdate(self, batchSize, learningRate, lmbda):
        """
            Based on the biases and weights gradients computed in self._delta and self._deltaBiases
        updates the weights and biases taking into account the learningRate and the regularization parameter lambda.
        :param batchSize: how many training samples were in this feedforward-backpropagation session
        :param learningRate: the learning rate of the algorithms
        :param lmbda: regularization parameter lambda
        :return: None
        """

        regParam = (learningRate*(lmbda/batchSize))
        self._weights = [(1 - learningRate * (lmbda / batchSize)) * w - (learningRate / batchSize) * nw
                        for w, nw in zip(self._weights, self._delta)]
        self._biases = [b - (learningRate / batchSize) * nb
                       for b, nb in zip(self._biases, self._deltaBias)]
        # for i in range(self._layerCount-1):
        #     #update the weights between layer i and layer i+1
        #     self._weights[i] = self._weights[i] + ((1/batchSize)*learningRate) * self._delta[i] + regParam*self._weights[i]
        #
        #     #update the biases of layer i+1
        #     self._biases[i] = self._biases[i] + ((1/batchSize)*learningRate) * self._deltaBias[i]



    def predict(self, x):
        """
            Predicts the output of x based on the network's weights.
        :param x: Input data. Must be in column shape, matching the input shape.
        :return: Column matrix representing the prediction.
        """
        return self._feedforward(x)


    def evaluate(self, x, y):
        pred = self.predict(x).T
        cost = self._costFunction.f(y,pred)
        return np.mean(cost)



    def fit(self, x, y, epochs=1,learningRate=0.01, lmbda=0.5,dropout=0.1 ,batchSize=16,validationData = None,validationRatio = 0.2, verbose=True):
        """
            Fits the neural network to the (x,y) dataset.

        :param x: Matrix containing on each line the independent variables of a training example.
        :param y: Column matrix containing the predictions for each training example.
        :param epochs: Number of epochs the algorithm has to run.
        :param learningRate: Learning rate of the algorithm.
        :param lmbda: Regularization parameter lambda.
        :param batchSize: How many training samples have to be processed at once
        :param validationRatio: In case :param validationData is None, a train/test split will be performed with validationRatio as test ratio;
            - Expecting a float number between 0 and 1
        :param validationData: [xValidation, yValidation] dataSet for validation. Can be None and the algorithm will split the dataset.
        :param verbose: If 'true', the algorithm will print the progress.
        :return: Returns the history for each epoch.
        """
        #parameter preprocessing
        x = np.asarray(x)
        y = np.asarray(y)

        if len(x) != len(y):
            raise NeuralNetworkException("Shapes {} and {} do not match.".format(x.shape,y.shape))
        if len(y.shape) == 1:
            y = np.asarray([y])
            y = y.reshape(len(x),1)
        if len(x.shape) == 1:
            l = len(x)
            x = np.asarray([x])
            x = x.reshape((l, self._inputShape[0]))


        if epochs < 1:
            epochs = 1
        if batchSize < 1:
            batchSize = 1
        if validationRatio < 0 or validationRatio >= 1:
            validationRatio = 0.2
        if validationData is None:
            x,xVal, y,yVal = train_test_split(x,y,test_size=validationRatio,random_state=0)
            validationData = (xVal,yVal)
        else:
            validationData[0],validationData[1] = np.asarray(validationData[0]),np.asarray(validationData[1])

        if len(y.shape) == 1:
            y = y.reshape((len(y),1))

        if verbose:
            return self._fitVerbose(x,y,epochs,learningRate,lmbda,dropout,batchSize,validationData)
        else:
            return self._fitQuiet(x,y,epochs,learningRate,lmbda,dropout,batchSize,validationData)

    def _fitVerbose(self, x, y, epochs, learningRate, lmbda,dropout, batchSize, validationData):
        """
            The verbose version of the fit function.
        """
        history = []
        print("Training on {} samples, validating on {}.".format(len(x), len(validationData[0])))
        for epoch in range(epochs):

            # code block for one epoch
            epochHistory = []

            xIndex = 0
            batchId = 0
            crtItems = min(batchSize, len(x) - xIndex)
            while crtItems > 0:
                # #train block for a batch of data
                regularizationAddition = .5 * (lmbda / crtItems) * sum(np.linalg.norm(w) ** 2 for w in self._weights)

                for i in range(xIndex, xIndex + crtItems):
                    xcrt, ycrt = x[i:i + 1], y[i:i + 1]

                    # feedforward xcrt
                    out = self._feedforward(xcrt,dropout)

                    # compute the error
                    grad = self._costFunction.deriv(np.reshape(ycrt,out.shape), out)
                    grad = grad + np.float64(regularizationAddition)

                    # perform backpropagation
                    self._backprop(grad)

                # after passing each example in the batch through the network, perform batch update
                self._batchUpdate(crtItems, learningRate, lmbda)
                self._resetGradients()

                # mark batchHistory
                evalMetric = self.evaluate(validationData[0], validationData[1])
                print("Batch {} / {}. Loss {}.".format(batchId,len(x)//batchSize+(len(x)%batchSize!=0),evalMetric), end="\r")

                # advance in the dataset
                xIndex += crtItems
                batchId += 1
                crtItems = min(batchSize, len(x) - xIndex)


            evalEpochTrain = self.evaluate(x,y)
            evalEpochValidation = self.evaluate(validationData[0],validationData[1])
            history.append({"trainLoss":evalEpochTrain, "validationLoss":evalEpochValidation})
            print("Epoch {}. Loss train {}; validation {}. ".format(epoch,evalEpochTrain,evalEpochValidation))

        return history



    def _fitQuiet(self, x, y, epochs, learningRate, lmbda,dropout, batchSize,validationData):
        """
            The quiet version of the fit function.
        """
        history = []

        for epoch in range(epochs):
            #code block for one epoch
            epochHistory = []

            xIndex = 0
            batchId = 0
            crtItems = min(batchSize, len(x)-xIndex)
            while crtItems > 0:
                # #train block for a batch of data
                regularizationAddition = self._castValue(
                    .5 * (lmbda / crtItems) * sum(np.linalg.norm(w) ** 2 for w in self._weights))

                for i in range(xIndex, xIndex + crtItems):
                    xcrt, ycrt = x[i:i + 1], y[i:i + 1]

                    # feedforward xcrt
                    out = self._feedforward(xcrt,dropout)

                    # compute the error
                    grad = self._costFunction.deriv(np.reshape(ycrt, out.shape), out)
                    grad = grad + np.float64(regularizationAddition)

                    # perform backpropagation
                    self._backprop(grad)

                # after passing each example in the batch through the network, perform batch update
                self._batchUpdate(crtItems, learningRate, lmbda)
                self._resetGradients()

                #advance in the dataset
                xIndex += crtItems
                batchId += 1
                crtItems = min(batchSize, len(x) - xIndex)

            evalEpochTrain = self.evaluate(x, y)
            evalEpochValidation = self.evaluate(validationData[0], validationData[1])
            history.append({"trainLoss": evalEpochTrain, "validationLoss": evalEpochValidation})

        return history


    def save(self, filePath):
        # TODO
        """
            Saves the model weights to the filePath provided.
        :param filePath: path to file
        :return:
        """
        pass

    def load(self, filePath):
        # TODO
        """
            Loads the model's weights from the file specified.
        :param filePath: path to file
        :return:
        """
        pass

    def _initWeights(self, weightRange):
        """
            Initializes the weight matrices from an uniform distribution in the given range.
        :param weightRange: tuple containing the lower and upper bounds of the weight value distribution.
        :return: list of weight matrices.
        """
        weights = []
        wrange = self._weightsRange
        for i in range(self._layerCount-1):
            #creating the weights between layers i and i+1
            size = (self._layerSizes[i+1], self._layerSizes[i])
            weights.append(np.random.uniform(low=wrange[0], high=wrange[1], size=size))
        return weights

    def _initActivationFunctions(self, activationFunction):
        """
            Sets the activation function list.

        :param activationFunction:
            1. Name of activation function. Will be used for all layers.
            2. List of activation function names. If not enough for each layer, the last will be used for the remaining layers.
        :raise NeuralNetworkException if format not valid or activation function not found
        :return: list of activation functions
        """
        #parse activation function argument
        if isinstance(activationFunction, list):
            if len(activationFunction) < self._layerCount - 1:
                activationFunction = activationFunction + [activationFunction[-1]]*(self._layerCount-1-len(activationFunction))
            elif len(activationFunction) > self._layerCount - 1:
                activationFunction = activationFunction[:self._layerCount-1]

        elif isinstance(activationFunction, str):
            activationFunction = [activationFunction]*(self._layerCount-1)
        else:
            raise NeuralNetworkException("{} not a valid format for an activation function.".format(activationFunction))

        #create activation functions for each function name
        result = []
        for name in activationFunction:
            func = self._activationFunctionMap.get(name)
            if func is None:
                raise NeuralNetworkException("{} not a known activation function name.".format(name))
            else:
                result.append(func)

        return result

    def _initCostFunction(self, costFunction):
        """
            Initializes the cost function.
        :param costFunction: one oh the option listed below
            1. Name of cost function.
            2. Class instance that implements CostFunction class
        :return: instance of the cost function
        """
        #parse cost function argument
        if isinstance(costFunction, str):
            func = self._costFunctonMap.get(costFunction)
            if func is None:
                raise NeuralNetworkException("Cost function {} not recognized.".format(func))
            else:
                return func
        elif isinstance(costFunction, CostFunctions):
            return costFunction

        else:
            raise NeuralNetworkException("Invalid argument for 'costFunction': {}.".format(costFunction))

    def _initBiases(self, biasRange):
        """
            Initializes the biases from a uniform distribution between biasRange[0] and biasRange[1]
        :param biasRange: tuple containing the lower and upper bounds of the bias value distribution.
        :return: Bias list for every column except input, matching column shapes.
        """
        biases = []

        for i in range(1,self._layerCount):
            shape = (self._layerSizes[i],1)
            biases.append(np.random.uniform(low=biasRange[0], high=biasRange[1], size=shape))

        return biases

    def _resetGradients(self):
        """
            Resets the sum of errors to 0, since the batch update has been done
        :return: None
        """
        self._delta = [np.zeros(weights.shape, dtype=float) for weights in self._weights]  # sums up the errors for each neuron connection for each layer
        self._deltaBias = [np.zeros((self._layerSizes[i], 1), dtype=float) for i in range(1, self._layerCount)]


    def _castValue(self,elem):
        return min(max(elem,-(1e8)),1e8)














