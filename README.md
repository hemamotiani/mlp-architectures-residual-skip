# Minimal MLP with Residual & Skip Connections

This project implements a custom Multi-Layer Perceptron (MLP) using the Keras Functional API with both **residual blocks** and **additional skip connections**. The model is trained to **perfectly overfit a single batch** of the UCI Covertype dataset, while intentionally performing poorly on unseen validation data — a controlled experiment in model capacity and generalization.

## 🧠 Key Features

- 📦 Built using **Keras Functional API**
- 🔁 Includes **custom residual blocks** with optional projection layers
- ➿ Implements **non-residual skip connections** across layers
- 📉 Trained to demonstrate **overfitting** behavior on a small batch
- 📊 Model architecture visualized using [Netron](https://netron.app/)

## 🗂️ Dataset

- **Source:** [UCI Covertype Dataset](https://archive.ics.uci.edu/dataset/31/covertype)
- **Classes:** 7 forest cover types
- **Features:** 54 cartographic variables (elevation, slope, soil types, etc.)
- **Preprocessing:**
  - Numerical features normalized with `StandardScaler`
  - Binary features (e.g., soil types) kept as-is
  - Multi-class label used as-is (no binarization)

## ⚙️ Architecture Overview

- Input: 54 features
- Residual block:
  - Dense → ReLU → Dense → Residual Add
  - Projection if dimensions mismatch
- Skip connection from early input layer to deep layer
- Final dense layers → softmax output for multi-class classification

> See `model_summary.png` or open `mlp_with_residuals.h5` in [Netron](https://netron.app/) for a full architecture diagram.

## 🧪 Training Setup

- Train only on **128 samples** from the dataset
- Stop when **training loss → 0**
- Evaluate on validation set to confirm poor generalization

```python
model.fit(X_batch, y_batch, epochs=500, verbose=0)
