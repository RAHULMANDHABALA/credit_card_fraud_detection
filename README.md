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

## Step 2: Applying Machine Learning Algorithms

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

### Evaluation Metrics Function
```python
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score

def metrics(actuals, predictions):
    print("Accuracy: {:.5f}".format(accuracy_score(actuals, predictions)))
    print("Precision: {:.5f}".format(precision_score(actuals, predictions)))
    print("Recall: {:.5f}".format(recall_score(actuals, predictions)))
    print("F1-score: {:.5f}".format(f1_score(actuals, predictions)))
```

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

### Final Model Evaluation
```python
rf_resampled = RandomForestClassifier(n_estimators=100)
rf_resampled.fit(train_X, train_Y)
predictions_resampled = rf_resampled.predict(test_X)

print("Evaluation of Random Forest Model After Oversampling")
metrics(test_Y, predictions_resampled.round())
```

## Key Findings
1. Initial models showed good accuracy but poor recall for fraud cases
2. SMOTE oversampling significantly improved fraud detection
3. Random Forest outperformed Decision Tree
4. Final model achieved better balance between precision and recall
```


