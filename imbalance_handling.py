import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
from collections import Counter 

# ------------------------- STEP-1: DATA LOADING AND INITIAL EXPLORATION----------------------

df = pd.read_csv('creditcard.csv')
print(df.head())
print(f"\nDataset shape: {df.shape}")
df.info()

print(df.isnull().sum())                        # Check for missing values

print(df['Class'].value_counts())
print(f"Fraudulent transactions: {df['Class'].value_counts()[1]} ({df['Class'].value_counts()[1]/df.shape[0]*100:.4f}%)")
print(f"Non-fraudulent transactions: {df['Class'].value_counts()[0]} ({df['Class'].value_counts()[0]/df.shape[0]*100:.4f}%)")

plt.figure(figsize=(6, 4))                      # Visualize original class distribution
sns.countplot(x='Class', data=df)
plt.title('Original Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.xlabel('Class')
plt.ylabel('Number of Transactions')
plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'])
plt.tight_layout()
plt.show()

# ------------------------- STEP-2: FEATURE SCALING ------------------------------

scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))          # Scale 'Amount'
df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))              # Scale 'Time'

df = df.drop(['Time', 'Amount'], axis=1)                                                # Drop original 'Time' and 'Amount' columns as we'll use their scaled versions
print(df.head())

# ------------------------- STEP-3: DATA SPLITTING ------------------------------

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

print("\nClass distribution in Training data before SMOTE:")
print(y_train.value_counts())
print("\nClass distribution in Testing data:")
print(y_test.value_counts())


# ------------------------- STEP-4: IMBALANCE HANDLING WITH SMOTE ------------------------------

# SMOTE (Synthetic Minority Over-sampling Technique) is applied ONLY to the training data. It's crucial NOT to apply SMOTE to the test set, as this would lead to data leakage

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nClass distribution in Training set AFTER SMOTE:")
print(Counter(y_train_resampled))
print(f"Resampled training set shape: X_train_resampled={X_train_resampled.shape}, y_train_resampled={y_train_resampled.shape}")

plt.figure(figsize=(6, 4))                      # Visualize class distribution after SMOTE
sns.countplot(x=y_train_resampled)
plt.title('Class Distribution in Training Set After SMOTE')
plt.xlabel('Class')
plt.ylabel('Number of Transactions')
plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'])
plt.tight_layout()
plt.show()

# ------------------------- STEP-5: SAVING PROCESSED DATA ------------------------------

X_train_resampled.to_csv('X_train_resampled.csv', index=False)
y_train_resampled.to_csv('y_train_resampled.csv', index=False, header=True)     # header=True to save column name
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False, header=True)                           # header=True to save column name


