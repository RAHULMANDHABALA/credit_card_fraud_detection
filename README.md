# Credit Card Fraud Detection Machine Learning Project

## Project Overview
In this machine learning project, we solve the problem of detecting credit card fraud transactions using:
- NumPy
- Scikit-learn
- Other Python libraries

We create a binary classifier and experiment with various machine learning techniques.

## Dataset Information
- 31 parameters (28 PCA-transformed features + Time + Amount)
- 284,807 total transactions (only 492 fraud cases)
- Severe class imbalance (99.83% genuine vs 0.17% fraud)

```python
import pandas as pd
dataframe = pd.read_csv("creditcard.csv")
dataframe.head()


## Step 1: Exploratory Data Analysis (EDA)

### Checking for Null Values
```python
dataframe.isnull().values.any()  # Returns False
```

### Analyzing Transaction Amounts
```python
dataframe["Amount"].describe()
```

### Class Distribution Analysis
```python
non_fraud = len(dataframe[dataframe.Class == 0])
fraud = len(dataframe[dataframe.Class == 1])
fraud_percent = (fraud / (fraud + non_fraud)) * 100
print("Number of Genuine transactions:", non_fraud)
print("Number of Fraud transactions:", fraud)
print("Percentage of Fraud transactions: {:.4f}".format(fraud_percent))
```


### Visualization
```python
import matplotlib.pyplot as plt
labels = ["Genuine", "Fraud"]
count_classes = dataframe.value_counts(dataframe['Class'], sort=True)
count_classes.plot(kind="bar", rot=0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()
```

![image](https://github.com/user-attachments/assets/c8ff3c35-d20d-45da-9ca4-11579e30ab5a)


### Data Preprocessing
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataframe["NormalizedAmount"] = scaler.fit_transform(dataframe["Amount"].values.reshape(-1, 1))
dataframe.drop(["Amount", "Time"], inplace=True, axis=1)
```

### Train-Test Split
```python
from sklearn.model_selection import train_test_split
Y = dataframe["Class"]
X = dataframe.drop(["Class"], axis=1)
(train_X, test_X, train_Y, test_Y) = train_test_split(X, Y, test_size=0.3, random_state=42)
```

![image](https://github.com/user-attachments/assets/994cd2e0-6a0b-4d13-9ba6-da93aa2abb5f)


## Step 2: Applying Machine Learning Algorithms

## Step 2: Apply Machine Learning Algorithms to Credit Card Dataset

We'll train different models on our dataset to observe which algorithm performs better for this binary classification problem (predicting fraud vs genuine transactions). Potential algorithms include:

- Random Forest
  ![image](https://github.com/user-attachments/assets/65d1e533-85a0-4817-bc55-508ede48b830)
- Decision Tree
  ![image](https://github.com/user-attachments/assets/3ad16c15-1b06-426b-8da2-9e37a949d62a)

- Support Vector Machines (SVM)
- Others

For this project, we'll focus on **Random Forest** and **Decision Tree** classifiers, then address class imbalance using the best-performing model.

### Algorithm Background

#### Decision Tree
A supervised machine learning algorithm used for both classification and regression. It builds a model that predicts the target variable by learning simple if-then-else decision rules from the training data.


![image](https://github.com/user-attachments/assets/65d1e533-85a0-4817-bc55-508ede48b830)


### Model Initialization
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier(n_estimators=100)
```

## Step 3: Model Training and Evaluation

### Training Models
```python
decision_tree.fit(train_X, train_Y)
random_forest.fit(train_X, train_Y)
```
![image](https://github.com/user-attachments/assets/33f27d73-af9a-4976-9bf2-57aed1cd6fee)

### Evaluation Metrics Function
```python
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score

def metrics(actuals, predictions):
    print("Accuracy: {:.5f}".format(accuracy_score(actuals, predictions)))
    print("Precision: {:.5f}".format(precision_score(actuals, predictions)))
    print("Recall: {:.5f}".format(recall_score(actuals, predictions)))
    print("F1-score: {:.5f}".format(f1_score(actuals, predictions)))
```
![image](https://github.com/user-attachments/assets/120630dd-0a59-4aff-af06-3bfc07faf9b2)
![image](https://github.com/user-attachments/assets/dc1b322e-d911-4491-909d-cb52b87148d0)


## Addressing Class Imbalance

### The Problem
Our Random Forest model outperforms Decision Trees, but the dataset suffers from severe class imbalance:
- **Genuine transactions**: >99%
- **Fraud transactions**: 0.17% 

Without addressing this imbalance, models will:
1. Prioritize the majority class (genuine transactions)
2. Achieve high accuracy but poor fraud detection
3. Fail to learn meaningful patterns for fraud cases

### Solution: Oversampling with SMOTE
We'll use the **Synthetic Minority Oversampling Technique (SMOTE)** to balance our dataset by:
- Generating synthetic fraud examples
- Creating a more balanced training distribution



### Addressing Class Imbalance with SMOTE
```python
from imblearn.over_sampling import SMOTE
from collections import Counter

X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
print("Resampled shape of X:", X_resampled.shape)
print("Resampled shape of Y:", Y_resampled.shape)
print("Class counts:", Counter(Y_resampled))

(train_X, test_X, train_Y, test_Y) = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)
```
![image](https://github.com/user-attachments/assets/327fae91-8aad-4355-9739-a158c172c07b)

### Final Model Evaluation
```python
rf_resampled = RandomForestClassifier(n_estimators=100)
rf_resampled.fit(train_X, train_Y)
predictions_resampled = rf_resampled.predict(test_X)

print("Evaluation of Random Forest Model After Oversampling")
metrics(test_Y, predictions_resampled.round())
```
![image](https://github.com/user-attachments/assets/a91a9876-bcde-4d83-a511-75e32544cfde)


## Key Findings
1. Initial models showed good accuracy but poor recall for fraud cases
2. SMOTE oversampling significantly improved fraud detection
3. Random Forest outperformed Decision Tree
4. Final model achieved better balance between precision and recall


