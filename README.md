# 🛡️ Cyber Attack Detection System

### *AI-Driven Threat Intelligence using Deep Neural Networks on the BETH Dataset*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-NumPy-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=for-the-badge)

---

## 📋 Table of Contents
- [Executive Summary](#-executive-summary)
- [Live Demo](#-live-demo)
- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [Technical Architecture](#-technical-architecture)
- [Performance Metrics](#-performance-metrics)
- [Business Impact](#-business-impact)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Author](#-author)

---

## 🎯 Executive Summary

In an era of sophisticated cyber threats, traditional signature-based detection systems are failing. This project implements a **custom Deep Neural Network (DNN)** built from scratch using NumPy to detect malicious activity in system logs. 

Trained on the **BETH Dataset**, the model achieves a **94.57% accuracy** and an outstanding **ROC-AUC of 0.9510**, effectively identifying threats in a highly imbalanced environment (600:1 ratio of benign to malicious events).

**Key Achievement:** Successfully detected **161,287 malicious events** while maintaining a false alarm rate of less than **0.6%**.

---

## 🚀 Live Demo

| Resource | Link |
| :--- | :--- |
| **🌐 Live Web Application** | [Streamlit Dashboard](https://bigtime5-cyber-attack-detection-system-app-7zrq38.streamlit.app/) |
| **📊 Analysis Report** | [HTML Report](https://bigtime5.github.io/Cyber-Attack-Detection-System/) |
| **📁 Source Code** | [GitHub Repo](https://github.com/BigTime5/Cyber-Attack-Detection-System) |

---

## 🚨 The Problem

Modern organizations face a relentless barrage of cyber threats:
*   **Sophistication:** Attacks evolve faster than traditional rule-based signatures can update.
*   **Volume:** Analysts are overwhelmed by millions of log entries daily.
*   **Imbalance:** Malicious events are needles in a haystack of benign traffic (often <1% of data).

The BETH dataset simulates this reality, presenting system event logs where malicious behavior is rare but devastating.

---

## 💡 The Solution

This project leverages a **Deep Learning approach** to learn complex patterns of malicious behavior directly from raw process data.

### Why This Model is Exceptional:
1.  **Zero-Dependency Architecture:** The neural network is built entirely with **NumPy matrix operations**, avoiding heavy frameworks like TensorFlow/PyTorch for a lightweight, transparent implementation.
2.  **Class Imbalance Handling:** Utilizes **Balanced Mini-Batch Gradient Descent** to force the model to learn minority class (malicious) patterns effectively.
3.  **Feature Engineering:** Incorporates polynomial interaction terms (e.g., `userId * processId`) to capture complex behavioral relationships.

---

## ⚙️ Technical Architecture

### Data Pipeline
*   **Input:** 7 raw features (`processId`, `threadId`, `parentProcessId`, `userId`, `mountNamespace`, `argsNum`, `returnValue`)
*   **Preprocessing:** Standard Scalar Normalization + Feature Augmentation
*   **Target:** Binary Classification (`sus_label`: 0=Benign, 1=Malicious)

### Neural Network Design
A 4-layer Fully Connected Network:

| Layer | Neurons | Activation | Purpose |
| :--- | :--- | :--- | :--- |
| **Input** | 9 Features | - | Augmented Feature Vector |
| **Hidden 1** | 64 | ReLU | Pattern Extraction |
| **Hidden 2** | 32 | ReLU | Feature Abstraction |
| **Hidden 3** | 16 | ReLU | High-Level Representation |
| **Output** | 1 | Sigmoid | Probability Calculation |

### Training Strategy
*   **Optimizer:** Gradient Descent (LR: 0.01)
*   **Loss:** Implicit Binary Cross-Entropy
*   **Epochs:** 30
*   **Batch Size:** 4096 (Balanced Sampling)

---

## 📊 Performance Metrics

The model demonstrates exceptional discriminative ability, balancing high detection rates with low false alarms.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | `94.57%` | High overall correctness |
| **Precision** | `0.9994` | Almost zero false positives |
| **Recall (Detection Rate)**| `0.9407` | Catches 94% of attacks |
| **F1-Score** | `0.9691` | Excellent balance |
| **ROC-AUC** | `0.9510` | Strong class separation |

### Confusion Matrix Analysis
*   **True Positives:** 161,287 (Attacks Detected)
*   **False Negatives:** 10,172 (Attacks Missed)
*   **False Positives:** 97 (False Alarms)
*   **True Negatives:** 17,411

### Feature Importance
The model identified **`userId`** (Correlation: 0.8567) as the most predictive indicator of malicious activity, followed by `processId` and `threadId`.

---

## 💰 Business Impact

Based on the simulation using industry standard cost metrics (IBM 2023):

*   **Average Breach Cost:** $4.24 Million
*   **Attacks Detected:** 161,287
*   **Estimated Savings:** `$478.6 Billion` (Risk Mitigation Value)
*   **False Alarm Cost:** `$4.8 Million` (Operational Cost)
*   **Net Benefit:** Massive ROI by preventing catastrophic data breaches.

> *Note: Simulation based on proportional scaling of dataset events to real-world incident costs.*

---

## 📁 Repository Structure
