# Load the holdout results from the pickle file
with open('pkl/holdout_results.pkl', 'rb') as file:
    holdout_results = pickle.load(file)

# Get scores for Decision Tree and SVM
decision_tree_score = holdout_results['decision_tree_score']
svm_score = holdout_results['svm_score']

# Generate confusion matrix and ROC curve subplots for Decision Tree
dt_pred = decision_tree.predict(X_val)
dt_cm = confusion_matrix(y_val, dt_pred)

# Generate confusion matrix for SVM
svm_pred = svm.predict(X_val)
svm_cm = confusion_matrix(y_val, svm_pred)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot Decision Tree Confusion Matrix
sns.heatmap(dt_cm, annot=True, fmt="d", ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix - Decision Tree (Holdout Method)")

# Print Decision Tree Classification Report
dt_report = classification_report(y_val, dt_pred)
print("Classification Report - Decision Tree (Holdout Method):\n", dt_report)

# Plot Decision Tree ROC Curve
dt_fpr, dt_tpr, _ = roc_curve(y_val, dt_pred)
dt_roc_auc = auc(dt_fpr, dt_tpr)
axes[0, 1].plot(dt_fpr, dt_tpr, label=f'AUC = {dt_roc_auc:.2f}')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_title("ROC Curve - Decision Tree (Holdout Method)")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend(loc="lower right")

# Print SVM Classification Report
svm_report = classification_report(y_val, svm_pred)
print("Classification Report - SVM (Holdout Method):\n", svm_report)

# Plot SVM Confusion Matrix
sns.heatmap(svm_cm, annot=True, fmt="d", ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix - SVM (Holdout Method)")

# Plot SVM ROC Curve
svm_fpr, svm_tpr, _ = roc_curve(y_val, svm_pred)
svm_roc_auc = auc(svm_fpr, svm_tpr)
axes[1, 1].plot(svm_fpr, svm_tpr, label=f'AUC = {svm_roc_auc:.2f}')
axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1, 1].set_title("ROC Curve - SVM (Holdout Method)")
axes[1, 1].set_xlabel("False Positive Rate")
axes[1, 1].set_ylabel("True Positive Rate")
axes[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.show()