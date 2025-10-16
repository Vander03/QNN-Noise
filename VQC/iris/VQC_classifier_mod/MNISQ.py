import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

ds= qml.data.load("mnisq")
print(f"Dataset: {ds}\n")
print(f"Circuits Selected: {ds[0].fidelity}")

# for i in ds:
#     print(i)

# @qml.qnode(qml.device("default.qubit"))
# def circuit():
#     for op in ds.circuits[0]:
#         qml.apply(op)

#     return qml.state()

# image_array = np.reshape(np.abs(circuit()[:784]), [28,28])
# #show the encoded image
# plt.imshow(image_array)