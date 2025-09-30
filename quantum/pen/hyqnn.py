import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from pennylane_cirq import ops as cirq_ops

dev = qml.device("cirq.mixedsimulator", wires=2, shots=5000)

# CHSH observables
PZ = qml.PauliZ(0)
PX = qml.PauliX(0)
H1 = qml.Hermitian(np.array([[1, 1], [1, -1]]) / np.sqrt(2), wires=1) # Hadamard gate matrix
H2 = qml.Hermitian(np.array([[1, -1], [-1, -1]]) / np.sqrt(2), wires=1) # orthogonal hadamard

# make bell pair function, takes noise as function
def bell_pair_with_noise(noise_gate, p):
    # qml.Hadamard(0)
    # qml.CNOT([0, 1])
    # apply to both wires
    qml.CNOT([0, 1])
    noise_gate(p, wires=0) 
    # noise_gate(p, wires=1)


# HyQNN Noise Circuit
def block(noise_gate, p):
    qml.CNOT([0, 1])
    noise_gate(p, wires=0) 
    noise_gate(p, wires=1)
        

# measurement circuits
def make_measurement(observ, noise_gate):
    @qml.qnode(dev)
    # p = noise vals
    def circuit(p):
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        # bell_pair_with_noise(noise_gate, p)
        for k in range(2):
            block(noise_gate, p)
        return qml.expval(observ)
    return circuit

# Build four correlator circuits per noise gate
def run_noise(noise_gate, noise_name, noise_vals):
    # curry functions
    circuits = [
        make_measurement(PZ @ H1, noise_gate),
        make_measurement(PZ @ H2, noise_gate),
        make_measurement(PX @ H1, noise_gate),
        make_measurement(PX @ H2, noise_gate),
    ]
    # pass noise vals into curried functions
    expvals = np.array([[c(p) for c in circuits] for p in noise_vals])
    # calculate CHSH
    S = expvals[:,0] + expvals[:,1] + expvals[:,2] - expvals[:,3]
    return S

noise_vals = np.linspace(0,1,25)

# List of noise channels you want
noise_channels = {
    "Phase Flip": cirq_ops.PhaseFlip,
    "Bit Flip": cirq_ops.BitFlip,
    "Depolarize": cirq_ops.Depolarize,
    "Phase Damp": cirq_ops.PhaseDamp,
    "Amplitude Damp": cirq_ops.AmplitudeDamp,
    # "Phase Flip": qml.PhaseFlip,
    # "Bit Flip": qml.BitFlip,
    # "Depolarize": qml.DepolarizingChannel,
    # "Phase Damp": qml.PhaseDamping,
    # "Amplitude Damp": qml.AmplitudeDamping,
}

plt.figure()
for label, gate in noise_channels.items():
    S = run_noise(gate, label, noise_vals)
    plt.plot(noise_vals, S, label=label)

plt.axhline(2, color="red", linestyle="--", label="Classical bound")
plt.axhline(2*np.sqrt(2), color="green", linestyle="--", label="Tsirelson bound")
plt.xlabel("Noise parameter")
plt.ylabel("CHSH value $S$")
plt.xticks(np.linspace(0,1,11))
plt.grid(True, 'both', 'x')
plt.legend()
plt.show()