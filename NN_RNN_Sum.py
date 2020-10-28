import copy
import numpy as np

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def sigmoidPrime(z):
    return z * (1-z)

class IntToBinary:
    int_to_binary = {}
    binary_dim = 8
    max_val = (2**binary_dim)
    binary_val = np.unpackbits(np.array([range(max_val)], dtype=np.uint8).T, axis=1)
    def __init__(self, binary_dim = 8):
        self.binary_dim = binary_dim
        for i in range(self.max_val):
            self.int_to_binary[i] = self.binary_val[i]

#constants
learning_rate = 0.1
inputLayerSize = 2
hiddenLayerSize = 16
outputLayerSize = 1
binary_dim = 8

#set binary converter
intToBinary = IntToBinary(binary_dim)
int_to_binary = intToBinary.int_to_binary
max_val = intToBinary.max_val

#set weights
W1 = 2 * np.random.random((inputLayerSize, hiddenLayerSize)) - 1
W2 = 2 * np.random.random((hiddenLayerSize, outputLayerSize)) - 1
W_h = 2 * np.random.random((hiddenLayerSize, hiddenLayerSize)) - 1
W1_update = np.zeros_like(W1)
W2_update = np.zeros_like(W2)
W_h_update = np.zeros_like(W_h)


for j in range(10000):
    #prepare training data
    a_int = np.random.randint(max_val/2)
    a = int_to_binary[a_int]
    b_int = np.random.randint(max_val/2)
    b = int_to_binary[b_int]
    c_int = a_int + b_int
    c = int_to_binary[c_int]
    d = np.zeros_like(c)

    overallError = 0
    output_layer_deltas = list()
    hidden_layer_values = list()
    hidden_layer_values.append(np.zeros(hiddenLayerSize))

    for position in range(binary_dim):
        #convert train data to binary
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T
        #train
        layer_1 = sigmoid(np.dot(X,W1) + np.dot(hidden_layer_values[-1],W_h))
        layer_2 = sigmoid(np.dot(layer_1, W2))
        #calculate error
        output_error = y - layer_2
        output_layer_deltas.append((output_error)*sigmoidPrime(layer_2))
        overallError += np.abs(output_error[0])
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        #
        hidden_layer_values.append(copy.deepcopy(layer_1))
    future_layer_1_delta = np.zeros(hiddenLayerSize)

    for position in range(binary_dim):
        #adjusment
        X = np.array([[a[position], b[position]]])
        layer_1 = hidden_layer_values[-position - 1]
        prev_hidden_layer = hidden_layer_values[-position-2]
        output_layer_delta = output_layer_deltas[-position-1]
        layer_1_delta = (future_layer_1_delta.dot(W_h.T) + output_layer_delta.dot(W2.T)) * sigmoidPrime(layer_1)
        W2_update += np.atleast_2d(layer_1).T.dot(output_layer_delta)
        W_h_update += np.atleast_2d(prev_hidden_layer).T.dot(layer_1_delta)
        W1_update += X.T.dot(layer_1_delta)
        future_layer_1_delta = layer_1_delta
        W1 += W1_update * learning_rate
        W2 += W2_update * learning_rate
        W_h += W_h_update * learning_rate
        W1_update *= 0
        W2_update *= 0
        W_h_update *= 0

    if ((j+1) % 1000 == 0):
        print(j+1)
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("\n")
