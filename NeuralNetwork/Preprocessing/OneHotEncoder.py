import numpy as np
def oneHotEncode(x):
    valueSet = sorted(list(set(x)))
    newArr = np.zeros(shape=(len(x),len(valueSet)))

    for i in range(len(x)):
        for j in range(len(valueSet)):
            if x[i] == valueSet[j]:
                newArr[i][j] = 1
                break
    return valueSet,newArr