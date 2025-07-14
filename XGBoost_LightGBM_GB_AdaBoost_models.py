import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier     # Base estimator for AdaBoost
from xgboost import XGBClassifier                   # eXtreme Gradient Boosting
from lightgbm import LGBMClassifier                 # Light Gradient Boosting Machine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, PrecisionRecallDisplay
import joblib 
import os 
from collections import Counter 

# -------------------------------- STEP-1: LOADING PROCESSED DATA --------------------------------------------

X_train_resampled = pd.read_csv('X_train_resampled.csv')
y_train_resampled = pd.read_csv('y_train_resampled.csv').squeeze() # .squeeze() to convert DataFrame to Series
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze() # .squeeze() to convert DataFrame to Series

print(f"Loaded resampled training set shape: {X_train_resampled.shape}, {y_train_resampled.shape}")
print(f"Loaded test set shape: {X_test.shape}, {y_test.shape}")
print(f"Class distribution in resampled training set: {Counter(y_train_resampled)}")
print(f"Class distribution in test set: {Counter(y_test)}")


# --------------------------- STEP-2: ADABOOST MODEL IMPLEMENTATION ----------------------------


print("\n--- Training AdaBoost Classifier ---")
# AdaBoost (Adaptive Boosting) trains a series of weak learners (typically Decision Stumps - max_depth=1) sequentially. Each subsequent learner focuses on the samples misclassified by the previous ones.
# n_estimators: The maximum number of estimators at which boosting is terminated.
# learning_rate: Weights applied to each classifier at each boosting iteration.

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42), # Base estimator (Decision Stump)
    n_estimators=100,                                               # Number of weak learners
    learning_rate=1.0,
    random_state=42
)

adaboost_model.fit(X_train_resampled, y_train_resampled)

# --------------------------- STEP-3: GRADIENT BOOSTING MODEL IMPLEMENTATION ----------------------------

print("\n--- Training Gradient Boosting Classifier ---")
# Gradient Boosting builds trees sequentially, where each tree tries to predict the residuals (errors) of the previous trees. It's more flexible than AdaBoost as it can use different loss functions.
# n_estimators: The number of boosting stages to perform.
# learning_rate: Shrinks the contribution of each tree.
# max_depth: Maximum depth of the individual regression estimators.

gradient_boosting_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gradient_boosting_model.fit(X_train_resampled, y_train_resampled)


# --------------------------- STEP-4: XGBOOST MODEL IMPLEMENTATION ----------------------------

print("\n--- Training XGBoost Classifier ---")

# XGBoost (eXtreme Gradient Boosting) is a highly optimized and popular gradient boosting implementation.
# It's known for its speed and performance, and includes regularization to prevent overfitting.
# n_estimators: Number of boosting rounds.
# learning_rate: Step size shrinkage to prevent overfitting.
# use_label_encoder=False and eval_metric='logloss' are for suppressing deprecation warnings.

xgboost_model = XGBClassifier(
    objective='binary:logistic', # For binary classification
    n_estimators=100,
    learning_rate=0.1,
    use_label_encoder=False,        # Suppress warning
    eval_metric='logloss',          # Suppress warning
    random_state=42,
    n_jobs=-1                       # Use all available CPU cores
)

xgboost_model.fit(X_train_resampled, y_train_resampled)


# --------------------------- STEP-5: LIGHTGBM MODEL IMPLEMENTATION ----------------------------


# LightGBM (Light Gradient Boosting Machine) is another highly efficient gradient boosting framework.
# It's often faster than XGBoost, especially on large datasets, due to its leaf-wise tree growth.
# n_estimators: Number of boosting rounds.
# learning_rate: Step size shrinkage.
# num_leaves: Max number of leaves in one tree.

lgbm_model = LGBMClassifier(
    objective='binary',             # For binary classification
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,                  # Default value
    random_state=42,
    n_jobs=-1                       # Use all available CPU cores
)

lgbm_model.fit(X_train_resampled, y_train_resampled)



# --------------------------- STEP-6: EVALUATION ----------------------------------------


def evaluate_model(model, X_test, y_test, model_name):      # Function to evaluate and print metrics (reused from bagging script)
    print(f"\n--- Evaluation for {model_name} ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]              # Probability of the positive class (fraud)

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

    metrics_dict = {                                        # Store metrics for comparison
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    }
    return metrics_dict, y_pred, y_prob, conf_matrix, class_report

# Evaluate all Boosting Classifiers
adaboost_metrics, y_pred_adaboost, y_prob_adaboost, cm_adaboost, cr_adaboost = evaluate_model(adaboost_model, X_test, y_test, "AdaBoost Classifier")
gb_metrics, y_pred_gb, y_prob_gb, cm_gb, cr_gb = evaluate_model(gradient_boosting_model, X_test, y_test, "Gradient Boosting Classifier")
xgb_metrics, y_pred_xgb, y_prob_xgb, cm_xgb, cr_xgb = evaluate_model(xgboost_model, X_test, y_test, "XGBoost Classifier")
lgbm_metrics, y_pred_lgbm, y_prob_lgbm, cm_lgbm, cr_lgbm = evaluate_model(lgbm_model, X_test, y_test, "LightGBM Classifier")


# Combine metrics into a DataFrame for easy comparison
all_metrics = pd.DataFrame([adaboost_metrics, gb_metrics, xgb_metrics, lgbm_metrics])
print("\n--- Comparison of Boosting Model Metrics ---")
print(all_metrics.set_index('Model'))


# --------------------------- STEP-7: VISUALIZATION ----------------------------------------

def plot_confusion_matrix_heatmap(cm, model_name, file_path):               # Function to plot Confusion Matrix (reused from bagging script)
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

def plot_precision_recall_curve(y_test, y_prob, model_name, file_path):     # Function to plot Precision-Recall Curve (reused from bagging script)
    plt.figure(figsize=(6, 5))
    display = PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=model_name)
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()

# Plot and save visualizations for each Boosting Classifier
plot_confusion_matrix_heatmap(cm_adaboost, "AdaBoost Classifier", os.path.join('results/boosting_models', 'adaboost_cm.png'))
plot_precision_recall_curve(y_test, y_prob_adaboost, "AdaBoost Classifier", os.path.join('results/boosting_models', 'adaboost_pr_curve.png'))

plot_confusion_matrix_heatmap(cm_gb, "Gradient Boosting Classifier", os.path.join('results/boosting_models', 'gradient_boosting_cm.png'))
plot_precision_recall_curve(y_test, y_prob_gb, "Gradient Boosting Classifier", os.path.join('results/boosting_models',  'gradient_boosting_pr_curve.png'))

plot_confusion_matrix_heatmap(cm_xgb, "XGBoost Classifier", os.path.join('results/boosting_models',  'xgboost_cm.png'))
plot_precision_recall_curve(y_test, y_prob_xgb, "XGBoost Classifier", os.path.join('results/boosting_models',  'xgboost_pr_curve.png'))

plot_confusion_matrix_heatmap(cm_lgbm, "LightGBM Classifier", os.path.join('results/boosting_models',  'lightgbm_cm.png'))
plot_precision_recall_curve(y_test, y_prob_lgbm, "LightGBM Classifier", os.path.join('results/boosting_models',  'lightgbm_pr_curve.png'))


# --------------------------- STEP-8: MODEL PERSISTENCE (SAVING MODELS) ----------------------------------------

# Save the trained models for future use
joblib.dump(adaboost_model, os.path.join('models', 'adaboost_classifier_model.joblib'))
joblib.dump(gradient_boosting_model, os.path.join('models', 'gradient_boosting_classifier_model.joblib'))
joblib.dump(xgboost_model, os.path.join('models', 'xgboost_classifier_model.joblib'))
joblib.dump(lgbm_model, os.path.join('models', 'lightgbm_classifier_model.joblib'))
