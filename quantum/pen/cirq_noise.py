import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from pennylane_cirq import ops as cirq_ops

dev = qml.device("cirq.mixedsimulator", wires=2, shots=1000)

# CHSH observables
PZ = qml.PauliZ(0)
PX = qml.PauliX(0)
H1 = qml.Hermitian(np.array([[1, 1], [1, -1]]) / np.sqrt(2), wires=1)
H2 = qml.Hermitian(np.array([[1, -1], [-1, -1]]) / np.sqrt(2), wires=1)
CHSH_observables = [PZ @ H1, PZ @ H2, PX @ H1, PX @ H2]


# subcircuit for creating an entangled pair of qubits
def bell_pair():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])


# circuits for measuring each distinct observable
@qml.qnode(dev)
def measure_PZH1():
    bell_pair()
    return qml.expval(PZ @ H1)


@qml.qnode(dev)
def measure_PZH2():
    bell_pair()
    return qml.expval(PZ @ H2)


@qml.qnode(dev)
def measure_PXH1():
    bell_pair()
    return qml.expval(PX @ H1)


@qml.qnode(dev)
def measure_PXH2():
    bell_pair()
    return qml.expval(PX @ H2)


# now we measure each circuit and construct the CHSH inequality
expvals = [measure_PZH1(), measure_PZH2(), measure_PXH1(), measure_PXH2()]

# The CHSH operator is PZ @ H1 + PZ @ H2 + PX @ H1 - PX @ H2
CHSH_expval = np.sum(expvals[:3]) - expvals[3]

noise_vals = np.linspace(0, 1, 25)

CHSH_vals = []
noisy_expvals = []

for p in noise_vals:
    # we overwrite the bell_pair() subcircuit to add
    # extra noisy channels after the entangled state is created
    def bell_pair():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        # cirq_ops.Depolarize(p, wires=0)
        cirq_ops.BitFlip(p, wires=0)
        cirq_ops.BitFlip(p, wires=1)
        # qml.CNOT(wires=[0, 1])
        # cirq_ops.PhaseDamp(p, wires=1)

    # measuring the circuits will now use the new noisy bell_pair() function
    expvals = [measure_PZH1(), measure_PZH2(), measure_PXH1(), measure_PXH2()]
    noisy_expvals.append(expvals)
noisy_expvals = np.array(noisy_expvals)
CHSH_expvals = np.sum(noisy_expvals[:, :3], axis=1) - noisy_expvals[:, 3]

print(qml.draw(measure_PZH1)())

# CHSH Calc
S = noisy_expvals[:,0] + noisy_expvals[:,1] + noisy_expvals[:,2] - noisy_expvals[:,3]

# Plot the individual observables
# First plot: all observables + CHSH

plt.figure()
plt.plot(noise_vals, noisy_expvals[:, 0], "D", label=r"$\hat{PZ}\otimes \hat{H1}$", markersize=5)
plt.plot(noise_vals, noisy_expvals[:, 1], "x", label=r"$\hat{PZ}\otimes \hat{H2}$", markersize=12)
plt.plot(noise_vals, noisy_expvals[:, 2], "+", label=r"$\hat{PX}\otimes \hat{H1}$", markersize=10)
plt.plot(noise_vals, noisy_expvals[:, 3], "v", label=r"$\hat{PX}\otimes \hat{H2}$", markersize=10)

plt.xlabel("Noise parameter")
plt.ylabel(r"Expectation value $\langle \hat{A}_i\otimes\hat{B}_j\rangle$")
plt.legend()

# Second plot: CHSH only, with bounds
plt.figure()
plt.plot(noise_vals, S, "o-", label="CHSH value", markersize=6)
plt.axhline(2, color="red", linestyle="--", label="Classical bound")
plt.axhline(-2, color="red", linestyle="--")
plt.axhline(2*np.sqrt(2), color="green", linestyle="--", label="Tsirelson bound")
plt.axhline(-2*np.sqrt(2), color="green", linestyle="--")

plt.xlabel("Noise parameter")
plt.ylabel("CHSH value S")
plt.legend()
plt.show()


fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, layout="constrained")

# First subplot: individual observables + CHSH
ax1.plot(noise_vals, S, "o-", label="CHSH value", markersize=6)
ax1.axhline(2, color="red", linestyle="--", label="Classical bound")
ax1.axhline(-2, color="red", linestyle="--")
ax1.axhline(2*np.sqrt(2), color="green", linestyle="--", label="Tsirelson bound")
ax1.axhline(-2*np.sqrt(2), color="green", linestyle="--")
ax1.set_xlabel("Noise parameter")
ax1.set_ylabel("CHSH value S")
ax1.legend()
ax0.legend()

# Second subplot: CHSH with bounds
ax1.plot(noise_vals, S, "o-", label="CHSH value", markersize=6)
ax1.axhline(2, color="red", linestyle="--", label="Classical bound")
ax1.axhline(-2, color="red", linestyle="--")
ax1.axhline(2*np.sqrt(2), color="green", linestyle="--", label="Tsirelson bound")
ax1.axhline(-2*np.sqrt(2), color="green", linestyle="--")
ax1.set_xlabel("Noise parameter")
ax1.set_ylabel("CHSH value S")
ax1.legend()

plt.show()

# from pennylane_cirq import ops as cirq_ops
# # Note that the 'Operation' op is a generic base class
# # from PennyLane core.
# # All other ops are provided by Cirq.
# available_ops = [op for op in dir(cirq_ops) if not op.startswith("_")]
# print("\n".join(available_ops))
