
---

# 👤 Facial Verification Siamese Network

> **A PyTorch-powered deep learning project for binary facial similarity matching.**

---

## 📌 Project Overview

This project implements a **Siamese Neural Network** designed to solve the facial verification problem. By processing pairs of images through twin convolutional branches, the model learns to calculate a "similarity score," effectively distinguishing between the same individual and different people.

### 📊 Dataset: LFW-a

* **Total Pairs:** 3,200
* **Input Resolution:** $105 \times 105$ pixels (Resized from $250 \times 250$)
* **Distribution:** * **Training:** 2,200 pairs (Balanced)
* **Testing:** 1,000 pairs (Balanced)



---

## 🏗️ Model Architecture

The network is built on a **Symmetric Twin CNN** architecture:

* **Feature Extraction:** 4 Convolutional stages (64 $\rightarrow$ 128 $\rightarrow$ 128 $\rightarrow$ 256 filters).
* **Normalization:** Integrated **Batch Normalization** and **Dropout** for robust training.
* **Decision Head:** L1 distance calculation followed by a **4096-unit Fully Connected layer** and Sigmoid output.

---

## 📈 Key Findings & Performance

Based on evaluation against the test set, the model demonstrated high discriminative power:

### **Quantitative Snapshot**

| Metric | Performance |
| --- | --- |
| **Highest Confidence (Match)** | **98.9%** |
| **Lowest Confidence (Non-Match)** | **0.0%** |
| **Optimization Strategy** | Grid Search (LR, Batch Size, Dropout) |

### **Qualitative Analysis**

* ✅ **High Robustness:** Successfully identified pairs with **98.9% confidence** even when shadows or camera angles obscured half of the face.
* ✅ **Clear Differentiation:** Consistently assigned a **0% score** to pairs with distinct physical features or skin tones.
* ⚠️ **Known Weakness:** The model occasionally flags **False Positives (~96% score)** when different individuals share very similar lighting conditions or facial silhouettes.

---

## 🚀 How to Use

1. **Environment:** Ensure you have `torch`, `torchvision`, and `PIL` installed.
2. **Data:** Place `lfwa.zip` in your root directory.
3. **Run:** Execute the `Facial_Verification_Siamese_Network.ipynb` notebook.
4. **Tuning:** Use the built-in **Grid Search** cell to find the optimal hyperparameters for your specific hardware.

---

## 🛠️ Tech Stack

* **Core:** PyTorch
* **Image Processing:** Pillow, OpenCV
* **Data Science:** NumPy, Matplotlib
* **Optimization:** Adam Optimizer, Early Stopping

---
