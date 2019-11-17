from abc import abstractmethod
import numpy as np

class CostFunction():
    @abstractmethod
    def f(self,act,pred):
        """
            Computes the cost between the act values and the predicted values.
        :param act: list of actual values
        :param pred: list of predicted values
        :return: the cost
        """
        raise NotImplementedError("Method 'f' not implemented.")

    @abstractmethod
    def deriv(self,act,pred):
        """
            Computes the cost between the act values and the predicted values.
        :param act: list of actual values
        :param pred: list of predicted values
        :return: the cost derivative
        """
        raise NotImplementedError("Method 'deriv' not implemented.")

class MSE(CostFunction):
    def f(self, act, pred):
        """
            Returns the mean squared error
        :param act:
        :param pred:
        :return:
        """
        diff = np.vectorize(self._limit)(act-pred)
        return np.vectorize(self._limit)(.5 * (diff**2))


    def _limit(self,elem):
        elem = max(elem, -(10**6))
        elem = min(elem, 10**6)
        return elem

    def deriv(self, act, pred):
        """
            Returns the derivative of the Mean Squared Error
        :param act:
        :param pred:
        :return:
        """
        act = np.asarray(act)
        pred = np.asarray(pred)
        diff = np.vectorize(self._limit)(act-pred)
        return diff

class CrossEntropy(CostFunction):
    def f(self, act, pred):
        # Cost = (labels * log(predictions) + (1 - labels) * log(1 - predictions)) / len(labels)
        return -act * np.log(pred) - (1 - act) * np.log(1 - pred)

    def deriv(self, act, pred):
        return pred-act