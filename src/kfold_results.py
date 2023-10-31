# Load the k-fold results from the pickle file
with open('pkl/kfold_results.pkl', 'rb') as file:
    results = pickle.load(file)
    decision_tree_scores = results['decision_tree_scores']
    svm_scores = results['svm_scores']

# Get the last fold data
fold = len(decision_tree_scores) - 1
dt_score = decision_tree_scores[fold]
svm_score = svm_scores[fold]

print(f"Fold {fold+1} - Decision Tree (J48) Score: {dt_score:.2f}")
print(f"Fold {fold+1} - SVM Score: {svm_score:.2f}")

# Generate confusion matrix and ROC curve subplots for Decision Tree
dt_pred = decision_tree.predict(X)
dt_cm = confusion_matrix(y, dt_pred)

# Generate confusion matrix for SVM
svm_pred = svm.predict(X)
svm_cm = confusion_matrix(y, svm_pred)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot Decision Tree Confusion Matrix
sns.heatmap(dt_cm, annot=True, fmt="d", ax=axes[0, 0])
axes[0, 0].set_title(
    f"Confusion Matrix - Decision Tree (kfold Method) (Fold {fold+1})")

# Print Decision Tree Classification Report
dt_report = classification_report(y, dt_pred)
print(f"Classification Report - Decision Tree (Fold {fold+1}):\n{dt_report}")

# Plot Decision Tree ROC Curve
dt_fpr, dt_tpr, _ = roc_curve(y, dt_pred)
dt_roc_auc = auc(dt_fpr, dt_tpr)
axes[0, 1].plot(dt_fpr, dt_tpr, label=f'AUC = {dt_roc_auc:.2f}')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_title(
    f"ROC Curve - Decision Tree (kfold Method) (Fold {fold+1})")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend(loc="lower right")

# Generate confusion matrix for SVM
svm_pred = svm.predict(X)
svm_cm = confusion_matrix(y, svm_pred)

# Plot SVM Confusion Matrix
sns.heatmap(svm_cm, annot=True, fmt="d", ax=axes[1, 0])
axes[1, 0].set_title(f"Confusion Matrix - SVM (kfold Method) (Fold {fold+1})")

# Print SVM Classification Report
svm_report = classification_report(y, svm_pred)
print(f"Classification Report - SVM (Fold {fold+1}):\n{svm_report}")

# Plot SVM ROC Curve
svm_fpr, svm_tpr, _ = roc_curve(y, svm_pred)
svm_roc_auc = auc(svm_fpr, svm_tpr)
axes[1, 1].plot(svm_fpr, svm_tpr, label=f'AUC = {svm_roc_auc:.2f}')
axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1, 1].set_title(f"ROC Curve - SVM (kfold Method) (Fold {fold+1})")
axes[1, 1].set_xlabel("False Positive Rate")
axes[1, 1].set_ylabel("True Positive Rate")
axes[1, 1].legend(loc="lower right")

plt.tight_layout()
plt.show()