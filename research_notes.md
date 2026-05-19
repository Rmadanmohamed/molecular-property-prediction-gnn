# Research Notes: GNN for Molecular Property Prediction

## 1. Graph Neural Networks (GNNs)

- **Concept:** GNNs are neural networks designed to operate directly on graph-structured data.
- **Mechanism:** They work by passing messages between nodes in the graph, aggregating information from neighbors to update node representations layer by layer.
- **Applications:** Social networks, recommendation systems, physics simulations, fake news detection, traffic prediction, and importantly, molecular chemistry (e.g., antibacterial discovery, property prediction).
- **Graph Components:** Nodes (entities, e.g., atoms), Edges (relations, e.g., bonds), Global attributes (graph-level properties).
- **Data Representation:** Information (features/embeddings) can be stored at nodes, edges, or globally.
- **Types of Tasks:**
    - **Graph-level:** Predict a single property for the entire graph (e.g., molecule's smell, binding affinity, or a quantum property like HOMO-LUMO gap). This is our target task type.
    - **Node-level:** Predict properties or roles of individual nodes (e.g., atom type, charge).
    - **Edge-level:** Predict properties or existence of edges (e.g., bond type, interaction strength).
- **Reference:** [A Gentle Introduction to Graph Neural Networks (Distill.pub)](https://distill.pub/2021/gnn-intro/)

## 2. Molecules as Graphs

- **Representation:** Molecules can be naturally represented as graphs where:
    - **Nodes:** Atoms
    - **Edges:** Covalent bonds
- **Node Features:** Atom type (e.g., C, H, O, N, F), charge, coordinates (spatial information).
- **Edge Features:** Bond type (single, double, triple), distance.

## 3. QM9 Dataset

- **Source:** Originally from "Quantum chemistry structures and properties of 134 kilo molecules" (Nature Scientific Data, 2014) and benchmarked in "MoleculeNet" (2018).
- **Content:** Contains ~134,000 stable small organic molecules (up to 9 heavy atoms: C, O, N, F, plus H).
- **Data:** Includes 3D spatial coordinates (low energy conformation) and 19 quantum chemical properties calculated using Density Functional Theory (DFT) at the B3LYP/6-31G(2df,p) level.
- **Properties (Targets):**
    - 0: Dipole moment (μ)
    - 1: Isotropic polarizability (α)
    - 2: Highest occupied molecular orbital energy (ε_HOMO)
    - 3: Lowest unoccupied molecular orbital energy (ε_LUMO)
    - 4: Gap between HOMO and LUMO (Δε)
    - 5: Electronic spatial extent (<R²>)
    - 6: Zero point vibrational energy (ZPVE)
    - 7: Internal energy at 0K (U₀)
    - 8: Internal energy at 298.15K (U)
    - 9: Enthalpy at 298.15K (H)
    - 10: Free energy at 298.15K (G)
    - 11: Heat capacity at 298.15K (c_v)
    - 12-15: Atomization energies/enthalpies/free energies at 0K and 298.15K
    - 16-18: Rotational constants (A, B, C)
- **PyTorch Geometric:** Provides a convenient interface (`torch_geometric.datasets.QM9`) to load the dataset, including pre-processed atom features based on "Neural Message Passing for Quantum Chemistry" (2017).
- **Reference:** [PyTorch Geometric QM9 Documentation](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.QM9.html)

## 4. PyTorch Geometric (PyG)

- **Library:** A geometric deep learning extension library for PyTorch.
- **Functionality:** Simplifies the implementation of GNNs by providing optimized layers, graph data handling utilities, and common benchmark datasets (like QM9).
- **Key Components:** `Data` objects (for single graphs), `Dataset` classes (like `QM9`), GNN layers (`GCNConv`, `GATConv`, `MessagePassing`), pooling operations, etc.

## 5. Proposed Project Idea

- **Task:** Graph-level regression.
- **Goal:** Predict the **HOMO-LUMO gap (Δε, target index 4)** for molecules in the QM9 dataset.
- **Method:** Implement a Graph Neural Network using PyTorch Geometric.
- **Rationale:** HOMO-LUMO gap is a fundamental quantum chemical property indicating molecular reactivity and electronic excitation energy. Predicting it accurately is a challenging and relevant task for GNNs.



## 6. GNN Architecture Design (SchNet for HOMO-LUMO Gap)

- **Model Choice:** SchNet (Schnett, K. T., et al. "SchNet–A deep learning architecture for molecules and materials." The Journal of Chemical Physics 148.24 (2018): 241705.)
    - **Reasoning:** Specifically designed for predicting quantum chemical properties of molecules, incorporates 3D structural information (interatomic distances) available in QM9, considered an advanced GNN model for this domain. PyTorch Geometric provides an implementation (`torch_geometric.nn.SchNet`).
- **Input:** QM9 molecular graphs. Node features will include atom types (embedded) and 3D positions (`pos`).
- **Output:** Single scalar value predicting the HOMO-LUMO gap (Δε, target index 4).
- **Core Architecture (`torch_geometric.nn.SchNet`):**
    - **Embedding Layer:** Maps atom types (integers) to continuous feature vectors.
    - **Interaction Blocks:** Multiple blocks performing message passing. Each block typically consists of:
        - Distance Calculation: Compute distances between atoms.
        - Radial Basis Function (RBF) Expansion: Expand distances using Gaussian basis functions.
        - Continuous-Filter Convolutional Layers (`SchNetConv`): Update atom embeddings by aggregating information from neighbors, weighted by filters derived from the RBF-expanded distances.
        - Shifted Softplus Activation: Often used within the interaction blocks.
    - **Pooling:** Global Add Pooling will be used to aggregate the final node embeddings into a single graph-level embedding.
- **Readout Head:**
    - A Multi-Layer Perceptron (MLP) applied to the pooled graph embedding.
    - Example: Linear -> ShiftedSoftplus -> Linear.
    - Output dimension: 1 (for the single target property).
- **Target Property:** HOMO-LUMO gap (index 4 in QM9 dataset's `y` tensor).
- **Loss Function:** Mean Absolute Error (MAE). This is common for QM9 benchmarks and less sensitive to outliers than MSE.
- **Optimizer:** Adam optimizer.
- **Training Details:**
    - Data Split: Standard QM9 splits (e.g., 110k train, 10k validation, ~13k test).
    - Batching: Use PyTorch Geometric's `DataLoader`.
    - Learning Rate Schedule: May be beneficial (e.g., reduce on plateau).

This design leverages the strengths of SchNet for molecular property prediction and aligns with common practices on the QM9 benchmark.
