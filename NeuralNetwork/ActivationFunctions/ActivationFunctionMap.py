import json
from NeuralNetwork.ActivationFunctions.ActivationFunctions import *

class ActivationFunctionMap:
    def __init__(self, configFile):
        self._file = configFile
        self._functions = self._config()

    def get(self, funcName):
        """
            If an activation function class called <funcName> exists, returns a class instance.
        :param funcName: name of activation function
        :return: class instance or None
        """
        funcCall = self._functions.get(funcName)
        if funcCall is None:
            return None
        else:
            return eval(funcCall)

    def _config(self):
        """
            Based on the JSON file provided in the constructor, builds a dictionary of key value pairs.
        :return: a dictionary with function names-function class instance code
        """
        functions = {}

        with open(self._file, 'r') as f:
            d = json.load(f)
            funcs = d['functions']
            for func in funcs:
                functions[func['name']] = func['classCall']

        return functions




