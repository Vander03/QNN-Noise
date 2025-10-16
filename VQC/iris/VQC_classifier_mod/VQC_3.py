
# --------------------------------------
# Testing Bed: Noisy pennylane classifer
# --------------------------------------
# Current state: Supervised classifier
# VERSION: 1.3
# Improvements:
# -> added pennylane provided state_prep function (more complex and well suited for simplification)

# TODO's
# --------------------------------------
# TODO: Adjust stateprep for more qubits, test with MNIST (currently very limited in Iris images)
# TODO: Adjust loss function to better suit larger images, explore contrastive when possible
# TODO: Display images after noisy circuits for visuals
# TODO: Adjust optimiser to suit larger images if needed


# COMMENTS:
# --------------------------------------
# -> Converves at low or extereme noise levels with epochs higher than HyQNN. Why was their epochs so limited
# -> Default pennylane noise channels suitable for testing, no change with Cirq channels but much slower than default


import pennylane as qml
# import qiskit
# import qiskit.providers.aer.noise as noise
from pennylane_cirq import ops as cirq_ops
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt

####################
# Params
epochs = 300 # number of epochs to train over
noise_params = [qml.BitFlip, qml.PhaseFlip, qml.PhaseDamping, qml.AmplitudeDamping, qml.DepolarizingChannel]
step_size = 0.01 # nesterov step size
num_qubits = 2 # number of wires in system
num_layers = 3 # number of VQC layers in system

# NOISE:
noise = qml.BitFlip
# noise = cirq_ops.BitFlip
noise_prob = 0.1

# TRAIN SETTINGS
opt = NesterovMomentumOptimizer(step_size)
batch_size = 5

# training graphs
train_acc = []
val_acc = []
loss = []

# quantum device for running circuits
# dev = qml.device("cirq.mixedsimulator")
dev3 = qml.device("default.mixed") # pennylane noise
# dev3 = qml.device("cirq.mixedsimulator", wires=2)


def layer(layer_weights):
    for wire in range(num_qubits):
        qml.Rot(*layer_weights[wire], wires=wire)
        # noise(noise_prob, wire) # apply noise with prob p to each wire after each gate
    qml.CNOT(wires=[0, 1])
    # noise(noise_prob, wire) # apply noise with prob p to each wire after each gate


# VQC structure
@qml.qnode(dev3)
def circuit(weights, x):
    qml.StatePrep(state=x, wires=range(num_qubits))

    for layer_weights in weights:
        layer(layer_weights)

    return qml.expval(qml.PauliZ(0))

# sum of output of circuit + trainable bias
def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

# supervised loss function (for now)
def square_loss(labels, predictions):
    # call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2) # square loss


# accuracy function
def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc


def cost(weights, bias, X, Y):
    # Transpose the batch of input data in order to make the indexing
    # in state_preparation work
    predictions = variational_classifier(weights, bias, X.T)
    return square_loss(Y, predictions)


# iris classication

def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])



data = np.loadtxt("variational_classifier/data/iris_classes1and2_scaled.txt")
X = data[:, 0:2] # collect data, slice away labels
print(f"First X sample (original)  : {X[0]}")

# pad the vectors to size 2^2=4 with constant values
padding = np.ones((len(X), 2)) * 0.1
X_pad = np.c_[X, padding]
print(f"First X sample (padded)    : {X_pad[0]}")

# normalize each input
normalization = np.sqrt(np.sum(X_pad**2, -1))
X_norm = (X_pad.T / normalization).T
print(f"First X sample (normalized): {X_norm[0]}")

# the angles for state preparation are the features
features = np.array([x for x in X_norm], requires_grad=False)
print(f"First features sample      : {features[0]}")

Y = data[:, -1] # collect labels

np.random.seed(0)
num_data = len(Y)
num_train = int(0.75 * num_data)
index = np.random.permutation(range(num_data))
features_train = features[index[:num_train]]
Y_train = Y[index[:num_train]]
feats_val = features[index[num_train:]]
Y_val = Y[index[num_train:]]

# We need these later for plotting
X_train = X[index[:num_train]]
X_val = X[index[num_train:]]

##############################################################################
# Optimization

weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

# train the variational classifier
weights = weights_init
bias = bias_init
for it in range(epochs):
    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    features_train_batch = features_train[batch_index]
    print(f"FEATURES: {features_train_batch.shape}")
    Y_train_batch = Y_train[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, features_train_batch, Y_train_batch)

    # Compute predictions on train and validation set
    predictions_train = np.sign(variational_classifier(weights, bias, features_train.T))
    predictions_val = np.sign(variational_classifier(weights, bias, feats_val.T))

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    if True: # changed from logging every 2nd one
        _cost = cost(weights, bias, features, Y)
        print(
            f"Iter: {it + 1:5d} | Cost: {_cost:0.7f} | "
            f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}"
        )
        train_acc.append(acc_train)
        val_acc.append(acc_val)
        loss.append(_cost)



# plot
epochs_logged = range(1, epochs + 1, 1) 

plt.figure(figsize=(10,6))
plt.ylim(0,1.1)
plt.plot(epochs_logged, train_acc, label="Train Accuracy")
plt.plot(epochs_logged, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Training and Validation Accuracy over Epochs. Noise:{noise}, Prob:{noise_prob}")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(epochs_logged, loss, label="Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss over Epochs. Noise:{noise}, Prob:{noise_prob}")
plt.legend()
plt.grid(True)
plt.show()
plt.figure()
cm = plt.cm.RdBu

# make data for decision regions
xx, yy = np.meshgrid(np.linspace(0.0, 1.5, 30), np.linspace(0.0, 1.5, 30))
X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]

# preprocess grid points like data inputs above
padding = 0.1 * np.ones((len(X_grid), 2))
X_grid = np.c_[X_grid, padding]  # pad each input
normalization = np.sqrt(np.sum(X_grid**2, -1))
X_grid = (X_grid.T / normalization).T  # normalize each input
features_grid = np.array([x for x in X_grid])  # angles are new features
predictions_grid = variational_classifier(weights, bias, features_grid.T)
Z = np.reshape(predictions_grid, xx.shape)

# plot decision regions
levels = np.arange(-1, 1.1, 0.1)
cnt = plt.contourf(xx, yy, Z, levels=levels, cmap=cm, alpha=0.8, extend="both")
plt.contour(xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,))
plt.colorbar(cnt, ticks=[-1, 0, 1])

# plot data
for color, label in zip(["b", "r"], [1, -1]):
    plot_x = X_train[:, 0][Y_train == label]
    plot_y = X_train[:, 1][Y_train == label]
    plt.scatter(plot_x, plot_y, c=color, marker="o", ec="k", label=f"class {label} train")
    plot_x = (X_val[:, 0][Y_val == label],)
    plot_y = (X_val[:, 1][Y_val == label],)
    plt.scatter(plot_x, plot_y, c=color, marker="^", ec="k", label=f"class {label} validation")

plt.legend()
plt.show()
