import numpy as np

class NeuralNetwork():
    def __init__(self):
        self.synaptic_weights = 2 * np.random.random((11, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def intToBitArray(self, training_inputs):
        return (((training_inputs[:, None] & (1 << np.arange(11)))) > 0).astype(float)

    def toBit(self, x):
        if(x>0.5):
            return 1
        else:
            return 0

    def train(self, training_iterations):
        for iteration in range(training_iterations):
            training_inputs = np.random.randint(1000, size=(2, 1))
            sum = training_inputs.sum()

            training_output = (((np.array([sum])[:,None] & (1 << np.arange(11)))) > 0).astype(float)
            output = self.fit(self.intToBitArray(training_inputs))
            out = (0 << np.arange(11)).astype(float)
            for i in range(len(output)):
                out[i] = self.toBit(output[i])


            output_val = out.dot(2**np.arange(out.size)[::-1])
            # back propagation
            error = sum - output_val
            error = (((np.array([int(error)]) & (1 << np.arange(11)))) > 0).astype(float)

            # performing weight adjustments
            adjustments = (self.sigmoid_derivative(output).T * error).T
            self.synaptic_weights += adjustments[:,None]

    def fit(self, inputs):
        output = np.dot(self.synaptic_weights,inputs)
        out = (0 << np.arange(11)).astype(float)
        for i in range(len(output)):
            out += output[i][0]
            out += output[i][1]
        out = self.sigmoid(out)
        return out

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random Weights: ")
    print(neural_network.synaptic_weights)

    neural_network.train(100000)

    print("Ending Weights: ")
    print(neural_network.synaptic_weights)

    inputs = training_inputs = np.random.randint(1000, size=(2, 1))
    print("Inputs: ", inputs)

    real_result = inputs.sum()
    res = neural_network.fit(neural_network.intToBitArray(inputs))
    neural_result = res.dot(2**np.arange(res.size)[::-1])

    print("real Result: ", real_result)
    print("neural Result: ", neural_result)
