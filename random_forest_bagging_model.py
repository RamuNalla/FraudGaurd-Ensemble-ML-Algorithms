import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier         
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, PrecisionRecallDisplay
import joblib 
import os 


# -------------------------------- STEP-1: LOADING PROCESSED DATA --------------------------------------------

X_train_resampled = pd.read_csv('X_train_resampled.csv')
y_train_resampled = pd.read_csv('y_train_resampled.csv').squeeze() # .squeeze() to convert DataFrame to Series
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze() # .squeeze() to convert DataFrame to Series

print(f"Loaded resampled training set shape: {X_train_resampled.shape}, {y_train_resampled.shape}")
print(f"Loaded test set shape: {X_test.shape}, {y_test.shape}")
print(f"Class distribution in resampled training set: {Counter(y_train_resampled)}")
print(f"Class distribution in test set: {Counter(y_test)}")


# --------------------------- STEP-2: BAGGING MODEL IMPLEMENTATION ----------------------------

# BaggingClassifier trains multiple base estimators (default is DecisionTreeClassifier)on random subsets of the original dataset (with replacement).
# n_estimators: The number of base estimators in the ensemble.
# max_samples: The number of samples to draw from X to train each base estimator.
# random_state: For reproducibility.

bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),      # Base estimator
    n_estimators=100,                                       # Number of trees
    max_samples=0.8,                                        # Use 80% of samples for each tree
    random_state=42,
    n_jobs=-1                                               # Use all available CPU cores for parallel training
)

bagging_model.fit(X_train_resampled, y_train_resampled)


# --------------------------- STEP-3: RANDOM FOREST IMPLEMENTATION ----------------------------

print("\n--- Training Random Forest Classifier ---")
# RandomForestClassifier is a specialized Bagging algorithm that introduces randomness in feature selection at each split.
# n_estimators: The number of trees in the forest.
# random_state: For reproducibility.

random_forest_model = RandomForestClassifier(
    n_estimators=100,                           # Number of trees
    random_state=42,
    n_jobs=-1                                   # Use all available CPU cores for parallel training
)

random_forest_model.fit(X_train_resampled, y_train_resampled)


# --------------------------- STEP-4: EVALUATION --------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):          # Function to evaluate and print metrics
    print(f"\n--- Evaluation for {model_name} ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]                  # Probability of the positive class (fraud)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)

    # Store metrics for comparison
    metrics_dict = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    }
    return metrics_dict, y_pred, y_prob, conf_matrix, class_report


bagging_metrics, y_pred_bagging, y_prob_bagging, cm_bagging, cr_bagging = evaluate_model(bagging_model, X_test, y_test, "Bagging Classifier")
rf_metrics, y_pred_rf, y_prob_rf, cm_rf, cr_rf = evaluate_model(random_forest_model, X_test, y_test, "Random Forest Classifier")

all_metrics = pd.DataFrame([bagging_metrics, rf_metrics])           # Combine metrics into a DataFrame for easy comparison
print(all_metrics.set_index('Model'))                               # Comparison of Bagging Model Metrics


# --------------------------- STEP-5: VISUALIZATION --------------------------------------------------


def plot_confusion_matrix_heatmap(cm, model_name, file_path):           # Function to plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Non-Fraud', 'Predicted Fraud'],
                yticklabels=['Actual Non-Fraud', 'Actual Fraud'])
    plt.title(f'Confusion Matrix ({model_name})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()


def plot_precision_recall_curve(y_test, y_prob, model_name, file_path): # Function to plot Precision-Recall Curve
    plt.figure(figsize=(6, 5))
    display = PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=model_name)
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()

# Plot and save visualizations for Bagging Classifier
plot_confusion_matrix_heatmap(cm_bagging, "Bagging Classifier", os.path.join('results/bagging_models', 'bagging_cm.png'))
plot_precision_recall_curve(y_test, y_prob_bagging, "Bagging Classifier", os.path.join('results/bagging_models', 'bagging_pr_curve.png'))

# Plot and save visualizations for Random Forest Classifier
plot_confusion_matrix_heatmap(cm_rf, "Random Forest Classifier", os.path.join('results/bagging_models', 'random_forest_cm.png'))
plot_precision_recall_curve(y_test, y_prob_rf, "Random Forest Classifier", os.path.join('results/bagging_models', 'random_forest_pr_curve.png'))


# --------------------------- STEP-6: MODEL PERSISTENCE (SAVING MODELS) --------------------------------------------------

bagging_model_path = os.path.join('models', 'bagging_classifier_model.joblib')
random_forest_model_path = os.path.join('models', 'random_forest_model.joblib')

joblib.dump(bagging_model, bagging_model_path)
joblib.dump(random_forest_model, random_forest_model_path)

