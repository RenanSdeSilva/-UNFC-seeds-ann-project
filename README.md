# Seeds Classification with Artificial Neural Networks  
### Final Machine Learning Project

## Project Overview
This project develops and evaluates a Feed-Forward Artificial Neural Network (ANN) to classify wheat seed varieties using the UCI Seeds Dataset.  
The dataset contains 210 samples, each with seven geometric measurements of wheat kernels, and the task is to assign each sample to one of three seed varieties.

The notebook includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and preparation
- ANN architecture design and implementation
- Training of two models with different dropout rates
- Hyperparameter comparison
- Final evaluation on the test set
- Confusion matrix analysis and interpretation

---

## Objectives
1. Conduct exploratory data analysis to examine feature distributions and class balance.  
2. Prepare the dataset through label encoding, train-test splitting, and feature scaling.  
3. Design a modular ANN with two hidden layers using TensorFlow/Keras.  
4. Compare the performance of two models with different dropout regularization strengths (0.2 vs. 0.5).  
5. Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.  
6. Select the best model based on test performance and interpret its classification behavior.

---

## Methodology

### 1. Data Loading and EDA
- The UCI Seeds Dataset (210 samples, 7 numerical features, 1 class label) was loaded and inspected.
- All features were confirmed to be numerical, with no missing values.
- The class distribution was examined and found to be perfectly balanced (70 samples per class).
- Summary statistics and a correlation matrix were used to understand feature relationships.

### 2. Data Preprocessing
- Class labels were recoded from {1, 2, 3} to {0, 1, 2}.
- A stratified 80/20 train-test split was applied to preserve class balance.
- StandardScaler was used to normalize features to zero mean and unit variance, with the scaler fit only on the training set.

### 3. Neural Network Architecture
A modular ANN builder function was created with the following structure:
- Two hidden layers (16 and 8 units)
- ReLU activation functions
- L2 regularization
- Dropout applied at configurable rates
- Softmax output layer for three-class classification
- Adam optimizer with a learning rate of 0.001

### 4. Training and Hyperparameter Comparison
Two models were trained under identical conditions (100 epochs, batch size 16):
- Model A: dropout rate of 0.2  
- Model B: dropout rate of 0.5

Training and validation accuracy and loss curves were compared to examine overfitting and generalization behavior.

### 5. Final Evaluation
- Both models were evaluated on the test set using accuracy, precision, recall, and F1-score.
- A confusion matrix was generated to analyze class-level performance.
- Model A (dropout = 0.2) demonstrated superior generalization, achieving strong accuracy and F1-scores across all classes.

---

## Execution Steps
1. Import required libraries and set up the environment.  
2. Load and explore the dataset.  
3. Apply preprocessing (encoding, scaling, train-test split).  
4. Define the ANN architecture using a modular function.  
5. Train Model A (dropout = 0.2).  
6. Train Model B (dropout = 0.5).  
7. Visualize training curves and compare performance.  
8. Evaluate both models on the test set.  
9. Analyze the confusion matrix of the best model.  
10. Summarize findings and select the final model.

---

## Final Outcome
- Model A (dropout 0.2) achieved the highest test accuracy and weighted F1-score.
- The model produced strong predictions across all three seed classes.
- Excessive dropout (0.5) led to underfitting, validating the chosen regularization level.
- The final selected model balances learning capability with generalization and provides reliable performance for this dataset.

---

## Dependencies
- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## License
Developed for academic purposes as part of the Final Machine Learning Project.
