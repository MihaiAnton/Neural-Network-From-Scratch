import numpy as np
from abc import abstractmethod

class ActivationFunction:

    def activate(self, x):
        """
            Calls the activation function for each subelement of x.
        :param x: numpyArray of numerals
        :return: list of activated elements
        """
        x = np.asarray(x)
        return self._f(x)

    def deriv(self, x):
        """
            Calls the gradient function for each subelement of x.
        :param x: numpyArray of numerals
        :return: list of derived elements
        """
        x = np.asarray(x)
        return self._d(x)

    @abstractmethod
    def _f(self,x):
        raise NotImplementedError("Activation function not implemented.")

    @abstractmethod
    def _d(self,x):
        raise NotImplementedError("Derivative function not implemented")

class LinearActivation(ActivationFunction):
    def _f(self, x):
        return x

    def _d(self, x):
        return 1

class ReluActivation(ActivationFunction):
    def __init__(self):
        self._f = np.vectorize(self._f)
        self._d = np.vectorize(self._d)

    def _f(self, x):
        if x <= 0:
            return 0
        else:
            return x



    def _d(self, x):
        if x <= 0:
            return 0
        else:
            return 1

class SigmoidActivation(ActivationFunction):
    def _f(self, x):
        return 1/(1+np.exp(-x))

    def _d(self, x):
        y = self._f(x)
        return (1-y)*y