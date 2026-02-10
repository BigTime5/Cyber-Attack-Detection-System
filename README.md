# 🛡️ Cyber Attack Detection with Deep Learning

### Advanced Threat Detection on the BETH Dataset using a Custom Neural Network

![Project Status](https://img.shields.io/badge/status-complete-green)
![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-brightgreen)
[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-red?style=for-the-badge&logo=streamlit)](https://bigtime5-cyber-attack-detection-system-app-7zrq38.streamlit.app/)

A deep learning project designed to proactively identify and classify malicious system events from benign ones. This model analyzes real-world simulated logs from the BETH dataset to enhance cybersecurity measures and protect organizations from sophisticated cyber threats.

---

## 🚀 Live Demo & Interactive Report

Experience the project's findings and interact with the model through the deployed web application and comprehensive analysis report.

- **[Interactive Streamlit Dashboard  streamlit.app/](https://bigtime5-cyber-attack-detection-system-app-7zrq38.streamlit.app/)**
- **[Full Analysis Report bigtime5.github.io](https://bigtime5.github.io/Cyber-Attack-Detection-System/)**

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Model Performance Highlights](#-model-performance-highlights)
- [Key Visualizations](#-key-visualizations)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [How to Run](#-how-to-run)
- [Key Findings & Business Impact](#-key-findings--business-impact)
- [Conclusion & Next Steps](#-conclusion--next-steps)
- [Contact](#-contact)
- [License](#-license)

---

## 🌐 Project Overview

Cyber threats are a growing concern for organizations worldwide, with attacks becoming more frequent and sophisticated. Traditional signature-based detection methods often fail to keep pace with new, evolving threats. This project addresses this challenge by leveraging a deep learning model to analyze system event logs.

The model is trained on the **BETH (Benchmark Environment for Threat Hunting)** dataset, which simulates real-world system behavior. By identifying subtle patterns in process IDs, user activity, and system calls, the neural network can distinguish between malicious (`sus_label=1`) and benign (`sus_label=0`) events with high accuracy, enabling proactive threat mitigation.

**Key Objectives:**
- **Design & Implement:** Create a robust deep learning model for binary classification of system events.
- **Handle Imbalance:** Address the severe class imbalance inherent in cybersecurity data (600:1 Benign:Malicious).
- **Feature Engineering:** Identify and engineer features that are highly predictive of malicious activity.
- **Deliver Insights:** Provide a comprehensive analysis of model performance, feature importance, and potential business impact.

---

## 📊 Model Performance Highlights

The trained neural network demonstrates exceptional performance in identifying malicious activities while maintaining a very low false alarm rate.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **94.57%** | Overall correctness across all events. |
| **Attack Detection Rate (Recall)** | **94.07%** | Successfully identified 94% of all malicious events. |
| **Precision** | **99.94%** | When the model predicts an attack, it's correct 99.9% of the time. |
| **False Alarm Rate** | **0.55%** | Only 0.55% of benign events were incorrectly flagged as malicious. |
| **ROC-AUC Score** | **0.9510** | Excellent ability to distinguish between malicious and benign classes. |
| **F1-Score (Malicious Class)** | **0.9691** | Strong balance between precision and recall. |

**Most Predictive Feature:** `userId` (Correlation: 0.8567)

---

## ✨ Key Visualizations

#### Confusion Matrix
This matrix visualizes the model's classification performance, detailing the trade-off between detecting attacks (True Positives) and avoiding false alarms (False Positives).


*The model correctly identified **161,287** malicious events while only misclassifying **97** benign events.*

#### Feature Importance
Analysis revealed that `userId` is by far the most significant predictor of malicious activity, highlighting the importance of monitoring user-level behavior.


*The high correlation of `userId` suggests that attacks in this dataset are strongly associated with specific user accounts.*

---

## 🛠️ Technology Stack

This project utilizes a range of modern data science and web development tools.

| Component | Technologies |
| :--- | :--- |
| **Data Analysis** | `Pandas`, `NumPy` |
| **Data Visualization** | `Matplotlib`, `Seaborn`, `Plotly` |
| **ML/Deep Learning** | `Scikit-learn`, Custom NumPy-based Neural Network |
| **Web Application** | `Streamlit` |
| **Development Environment** | `Jupyter Notebook` |

---

## 📂 Repository Structure

The repository is organized to separate data, source code, analysis, and application logic for clarity and reproducibility.
