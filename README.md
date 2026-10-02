# Logistic Regression from Scratch 🚀

## Overview
This repository contains a mini-project demonstrating the implementation of a Logistic Regression model entirely from scratch using Python. The primary goal of this project is to explore and understand the mathematical foundations of Machine Learning algorithms rather than relying on black-box functions from high-level libraries.

## Key Features
* **Symbolic Mathematics:** Used `SymPy` to define the Sigmoid function, its derivative, and mathematically derive the Gradient of the Binary Cross-Entropy Loss function.
* **Custom Optimization:** Implemented a custom Gradient Descent algorithm using `NumPy` to iteratively update weights and biases based on the derived gradients[cite: 2].
* **Evaluation Metrics from Scratch:** Extracted prediction probabilities to manually calculate the True Positive Rate (TPR) and False Positive Rate (FPR) across various thresholds[cite: 2].
* **AUC Calculation:** Computed the Area Under the Curve (AUC) manually using `np.trapezoid`, achieving a score of **0.9375**[cite: 2].
* **Validation:** Verified the accuracy of the manual implementation by comparing it against `sklearn.metrics.roc_auc_score`, resulting in a perfect match (0.9375)[cite: 2].

## Technologies Used
* **Python 3**[cite: 2]
* **NumPy:** For array operations, manual gradient descent, and trapezoidal integration[cite: 2].
* **SymPy:** For symbolic math and derivatives[cite: 2].
* **Matplotlib:** For visualizing the Cost function using contour plots[cite: 2].
* **Scikit-Learn:** Used exclusively to validate the manual AUC calculation[cite: 2].

## Project Structure
* `mini_loyiha_2month(homework).ipynb`: The main Jupyter Notebook containing all the mathematical proofs, algorithm implementations, and evaluations[cite: 2].

## How to Run
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
   ```bash
   pip install numpy sympy matplotlib scikit-learn