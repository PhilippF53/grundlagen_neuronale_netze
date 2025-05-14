import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

# Initial state
inputs = np.array([0.0, 0.0])
bias_input = 1.0

# Weight matrix
weights = np.array(
    [
        [-4.0, -1.5, -3.37],  # o1 = tanh(w11 * o1 + w12 * o2 + w_bias1)
        [1.5, 0.0, 0.125],  # o2 = tanh(w21 * o1 + w22 * o2 + w_bias2)
    ]
)


def tanh_with_bias(inputs, weights):
    extended_inputs = np.append(inputs, bias_input)
    return (2 / (1 + np.exp(-2 * np.dot(weights, extended_inputs)))) - 1


steps = 100
output = np.zeros((steps, 2))
inputs = np.array([0.0, 0.0])

for step in range(steps):
    output[step] = tanh_with_bias(inputs, weights)
    inputs = output[step]


plt.figure(figsize=(8, 5))
plt.plot(output[:, 0], output[:, 1], marker="o", markersize=5)
plt.xlabel("Neuron 1 Output")
plt.ylabel("Neuron 2 Output")
plt.show()
