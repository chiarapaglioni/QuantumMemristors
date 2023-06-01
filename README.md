# Quantized-HodgkinHuxleyModel

## Hodgkin Huxley Model

The package **HHModel** provides the implementation of the Quantized Single-Ion and Three-Ion Hodgkin-Huxley Model [1][2].

- Classes `QHH_1.py` and `QSim_1.py` provide the *implementation* and *simulation* of the Single-Ion version of the model respectively [1].
- Classes `QHH_3.py` and `QSim_3.py` provide the *implementation* and *simulation* of the Three-Ion version of the model respectively [2]. 


## Experimental Photonic Quantum Memristor

The package **memristor** provides the implementation of a Experimental photonic quantum memristor [3]. 
In particular, quantum memristors are used in a quantum reservoir computer that solves a classification problem based on the MINST database.

- `main.py` provides the general structure of the quantum reservoir computer.
- The data from the MINST database is encoded in the quantum domain through the class *QEncoder* in `encode.py`. 
- The encoded data is then passed through the quantum reservoir which is composed of quantum memristors. The implementation of the these components can be found in `memristor/utility.py`. 
- Packages `qinfo` and `ucell` provide useful functions for problems in quantum information theory that are used in the previous classes.

## Quantum Memristor

The package **q_memristor** provides the implementation of quantum circuits for memristive devices [4].

- `memristor_1t.py` provides the implementation of a circuit for a single-time-step quantum memristor simulation with three basic steps of initialization, evolution, and the measurement process.
- `memristor_coupled.py` is an extension of the previous circuit as it provides a dynamical simulation where the evolution step is repeated *n* times before the measurement process.
- `memristor_dynamic.py` provides the implementation of a native interaction between two coupled quantum memristors in a digital quantum circuit.


## References 

[1] Gonzalez-Raya, T., Cheng, X. H., Egusquiza, I. L., Chen, X., Sanz, M., & Solano, E. (2019). Quantized single-ion-channel Hodgkin-Huxley model for quantum neurons. Physical Review Applied, 12(1), 014037. https://doi.org/10.1103/PhysRevApplied.12.014037

[2] Gonzalez-Raya, T., Solano, E., & Sanz, M. (2020). Quantized three-ion-channel neuron model for neural action potentials. Quantum, 4, 224. https://doi.org/10.22331/q-2020-01-20-224

[3] Spagnolo, M., Morris, J., Piacentini, S., Antesberger, M., Massa, F., Crespi, A., ... & Walther, P. (2022). Experimental photonic quantum memristor. Nature Photonics, 16(4), 318-323. https://doi.org/10.1038/s41566-022-00973-5

[4] Guo, Y.-., Albarrán-Arriagada, F., Alaeian, H., Solano, E., & Barrios, G. (2021). Quantum Memristors with Quantum Computers. Physical Review Applied, 18(2). https://doi.org/10.1103/physrevapplied.18.024082