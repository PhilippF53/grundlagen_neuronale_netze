import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

# transferfunction
def tanh(inputs, weights, bias):
    return 2 / (1 + np.exp(-2 * (np.dot(inputs, weights) + bias))) - 1

# parameters
inputs = np.array([0.0, 0.0])
weights = np.array(
    [
        [-4.0, 1.5],
        [-1.5, 0.0],
    ]
)
bias = np.array([-3.37, 0.125])
steps = 50
inputs = np.array([0.0, 0.0])
output = np.zeros((steps, 2))

# generate outputs
for step in range(steps):
    output[step] = tanh(weights, inputs, bias)
    inputs = output[step]


plt.figure(figsize=(8, 5))
plt.plot(output[:, 0], output[:, 1], marker="o", markersize=5)
plt.xlabel("Neuron 1 Output")
plt.ylabel("Neuron 2 Output")
plt.show()
