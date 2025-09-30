import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


# Training + test data
X = np.linspace(0, 2*np.pi, 5)
X.requires_grad = False
Y = np.sin(X)

X_test = np.linspace(0.2, 2*np.pi+0.2, 20)  # more test points for smoother lines
Y_test = np.sin(X_test)

noise_levels = np.linspace(0,0.5,9)

def make_qnode(noise):
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def quantum_circuit(datapoint, params):
        # apply possible noise after every gate
        qml.RX(datapoint, wires=0)
        if noise > 0:
            qml.DepolarizingChannel(noise, wires=0)
        qml.Rot(params[0], params[1], params[2], wires=0)
        if noise > 0:
            qml.DepolarizingChannel(noise, wires=0)
        return qml.expval(qml.PauliY(0))

    return quantum_circuit

def loss_func(predictions, targets):
    return np.sum((np.array(predictions) - np.array(targets))**2)

def cost_fn(params, circuit):
    preds = [circuit(x, params) for x in X]
    return loss_func(preds, Y)

opt = qml.GradientDescentOptimizer(stepsize=0.3)

plt.figure(figsize=(8,6))

# Scatter actual data
plt.scatter(X, Y, c='b', marker="s", s=30, label="Train outputs")
plt.scatter(X_test, Y_test, c='r', marker="o", s=30, label="Test outputs")

for noise in noise_levels:
    circuit = make_qnode(noise)
    params = np.array([0.1,0.1,0.1], requires_grad=True)

    # Train
    for i in range(100):
        params, _ = opt.step_and_cost(lambda p: cost_fn(p, circuit), params)

    # Predictions
    preds = [circuit(x, params) for x in X_test]

    # Plot with a line connecting predictions
    plt.plot(X_test, preds, marker="x", label=f"Noise {noise}")
    # Visualize the circuit
    # fig, ax = qml.draw_mpl(circuit)(X_test[0], params)
    # fig.savefig('pennylane_circuit.png')

plt.xlabel("Inputs")
plt.ylabel("Outputs")
plt.title("QML sine approximation with noise")
plt.legend()
plt.show()