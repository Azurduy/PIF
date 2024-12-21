import numpy as np
import matplotlib.pyplot as plt

# Create an array of x values
x = np.linspace(-10, 10, 400)

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # For numerical stability
    return e_x / np.sum(e_x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def hard_sigmoid(x):
    return np.clip((x + 1) / 2, 0, 1)

# Plot each function
plt.figure(figsize=(12, 10))

# Sigmoid
plt.subplot(3, 3, 1)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid")

# Tanh
plt.subplot(3, 3, 2)
plt.plot(x, tanh(x))
plt.title("Tanh")

# ReLU
plt.subplot(3, 3, 3)

plt.plot(x, hard_sigmoid(x))
plt.title("Hard Sigmoid")

# Softmax
x_softmax = np.linspace(-10, 10, 400)
plt.subplot(3, 3, 4)
plt.plot(x_softmax, softmax(x_softmax))
plt.title("Softmax")

# ELU
plt.subplot(3, 3, 5)
plt.plot(x, elu(x))
plt.title("ELU")

# Hard Sigmoid
plt.subplot(3, 3, 6)
plt.plot(x, relu(x))
plt.title("ReLU")


plt.savefig("./inout/"+"activations.eps", format="eps", bbox_inches="tight", dpi=1200)
plt.tight_layout()
plt.show()
