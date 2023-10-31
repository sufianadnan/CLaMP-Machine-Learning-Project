# Load results from Leave-One-Out Cross-Validation
with open('pkl/loocv_results.pkl', 'rb') as file:
    loocv_results = pickle.load(file)

# Extract scores for Decision Tree and SVM for LOOCV
decision_tree_scores_loocv = loocv_results['decision_tree_scores']
svm_scores_loocv = loocv_results['svm_scores']

# Calculate ROC and other metrics for Decision Tree for Leave-One-Out Cross-Validation
fpr_dt_loocv, tpr_dt_loocv, _ = roc_curve(
    y, decision_tree.predict_proba(X)[:, 1])
roc_auc_dt_loocv = roc_auc_score(y, decision_tree.predict(X))

# Calculate ROC and other metrics for SVM for Leave-One-Out Cross-Validation
fpr_svm_loocv, tpr_svm_loocv, _ = roc_curve(y, svm.decision_function(X))
roc_auc_svm_loocv = roc_auc_score(y, svm.predict(X))

# Generate confusion matrix and ROC curve subplots for Decision Tree and SVM for LOOCV
decision_tree_pred = decision_tree.predict(X)
decision_tree_cm = confusion_matrix(y, decision_tree_pred)

svm_pred = svm.predict(X)
svm_cm = confusion_matrix(y, svm_pred)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot Decision Tree Confusion Matrix
sns.heatmap(decision_tree_cm, annot=True, fmt="d", ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix - Decision Tree (LOOCV)")

# Print Decision Tree Classification Report
dt_report = classification_report(y, decision_tree_pred)
print("Classification Report - Decision Tree (LOOCV):\n", dt_report)

# Plot Decision Tree ROC Curve
axes[0, 1].plot(fpr_dt_loocv, tpr_dt_loocv, color='darkorange', lw=2,
                label='ROC curve (area = %0.2f)' % roc_auc_dt_loocv)
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve - Decision Tree (LOOCV)')
axes[0, 1].legend(loc='lower right')

# Plot SVM Confusion Matrix
sns.heatmap(svm_cm, annot=True, fmt="d", ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix - SVM (LOOCV)")

# Print SVM Classification Report
svm_report = classification_report(y, svm_pred)
print("Classification Report - SVM (LOOCV):\n", svm_report)

# Plot SVM ROC Curve
axes[1, 1].plot(fpr_svm_loocv, tpr_svm_loocv, color='darkorange', lw=2,
                label='ROC curve (area = %0.2f)' % roc_auc_svm_loocv)
axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1, 1].set_xlim([0.0, 1.0])
axes[1, 1].set_ylim([0.0, 1.05])
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve - SVM (LOOCV)')
axes[1, 1].legend(loc='lower right')

plt.tight_layout()
plt.show()