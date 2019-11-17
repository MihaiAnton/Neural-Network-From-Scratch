class NeuralNetworkException(BaseException):
    def __init__(self, message):
        super(NeuralNetworkException, self).__init__(message)
        self._message = message

    def getError(self):
        return self._message