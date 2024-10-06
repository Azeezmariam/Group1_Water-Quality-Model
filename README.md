# Water Quality Model - Group 1

## Project Overview

This repository contains the implementation of a machine learning model designed to classify water potability based on various chemical and physical properties. The model uses regularization and optimization techniques to improve performance and avoid overfitting. The project explores vanilla model implementation and comparisons of L1 and L2 regularization techniques, with contextual explanations of their impact on model performance.

---

## Table of Contents

1. [Team Members and Roles](#team-members-and-roles)
2. [Introduction](#introduction)
3. [Dataset Overview](#dataset-overview)
4. [Data Preparation](#data-preparation)
5. [Model Implementation](#model-implementation)
6. [Regularization Techniques](#regularization-techniques)
    - L1 Regularization
    - L2 Regularization
7. [Comparison of L1 vs L2](#comparison-of-l1-vs-l2)
8. [Error Analysis](#error-analysis)
9. [Final Model Evaluation](#final-model-evaluation)
10. [Conclusion](#conclusion)
11. [How to Run](#how-to-run)
12. [Dependencies](#dependencies)
13. [References](#references)

---

## Team Members and Roles

- **Data Handler**: [Mariam Azeez](https://github.com/azeezmariam/)
- **Vanilla Model Implementor**: Charite Uwatwembi
- **Model Optimizer 1 (L1 Regularization)**: Daniel Ndungu
- **Model Optimizer 2 (L2 Regularization)**: Mohamed Yasin

---

## Introduction

This project focuses on building a neural network to predict whether water is potable based on chemical and physical parameters. The model evaluates various optimization techniques, comparing the performance of L1 and L2 regularization to mitigate overfitting and enhance generalization.

---

## Dataset Overview

The dataset contains the following features:
- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity
- Potability (Target Variable)

The dataset was cleaned to address missing values by filling them with the mean or median of their respective columns.

---

## Data Preparation

1. **Missing Values**: Filled missing data with mean or median values.
2. **Scaling**: Features were standardized using `StandardScaler` to ensure uniform data ranges.
3. **Splitting**: The dataset was split into 80% training and 20% testing sets.

---

## Model Implementation

### Vanilla Model

- The model architecture consists of:
  - Input layer with 64 units and ReLU activation.
  - Hidden layer with 32 units and ReLU activation.
  - Output layer with sigmoid activation for binary classification.
- Optimized using Adam with binary cross-entropy as the loss function.
- The model was trained for 100 epochs, achieving around **65.09%** accuracy on the test set.

---

## Regularization Techniques

### L1 Regularization

- L1 regularization encourages sparsity by shrinking weights toward zero, effectively performing feature selection.
- The model was trained using the `RMSprop` optimizer with a learning rate of 0.01.
- **Test accuracy**: 62.80%
  
### L2 Regularization

- L2 regularization prevents large weights by penalizing them, ensuring smoother generalization across all features.
- The Adam optimizer was used, with early stopping applied to avoid overfitting.
- **Test accuracy**: 64.90%

---

## Comparison of L1 vs L2

- **L1 Regularization** encourages the model to prioritize only the most important features, shrinking the weights of less relevant features to zero. This can be useful when you expect sparsity in your features but may result in lower accuracy if too many features contribute to the prediction.

- **L2 Regularization** spreads the weight evenly across all features, preventing any single feature from dominating the model. In the water quality dataset, L2 was more effective as it captured broader relationships between features.

- **Contextual Explanation**: In this dataset, L2's higher performance suggests that the feature relationships are more distributed and interdependent, making L2 regularization a better fit for balancing the model's complexity without neglecting important attributes.

---

## Error Analysis

- Confusion matrices were generated for both L1 and L2 models.
- Both models struggled with false positives and false negatives, likely due to class imbalance.
- Class weights were computed and used to mitigate the effect of imbalanced classes.

---

## Final Model Evaluation

The best-performing model utilized **L2 regularization** with dropout layers. This model achieved **63.0%** accuracy, and the combination of L2 regularization and dropout effectively prevented overfitting, improving the model’s generalization on unseen data.

---

## Conclusion

The project successfully demonstrated how L1 and L2 regularization techniques influence model performance. L2 regularization, particularly with dropout, performed better, showing the importance of careful regularization in neural networks. The next steps include further tuning of model hyperparameters and addressing class imbalance for better performance.

---

## How to Run

1. Clone this repository:
   ```
   git clone https://github.com/Azeezmariam/Group1_Water-Quality-Model
   ```
2. Install the required dependencies (see [Dependencies](#dependencies)).
3. Download the dataset and place it in the appropriate directory.
4. Run the notebook or Python script:
   ```
   python water_quality_model.py
   ```

---

## Dependencies

Ensure the following dependencies are installed:
- TensorFlow
- Keras
- Numpy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

Install the dependencies using:
```
pip install -r requirements.txt
```

---

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Regularization](https://keras.io/api/layers/regularizers/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

=======
