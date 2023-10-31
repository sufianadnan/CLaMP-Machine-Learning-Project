# Load results from Holdout Method
with open('pkl/holdout_results.pkl', 'rb') as file:
    holdout_results = pickle.load(file)

# Load results from k-fold Cross-Validation
with open('pkl/kfold_results.pkl', 'rb') as file:
    kfold_results = pickle.load(file)

# Load results from Leave-One-Out Cross-Validation
with open('pkl/loocv_results.pkl', 'rb') as file:
    loocv_results = pickle.load(file)

# Extract scores for Decision Tree and SVM from different methods
decision_tree_scores_holdout = holdout_results['decision_tree_score']
decision_tree_scores_kfold = kfold_results['decision_tree_scores']
decision_tree_scores_loocv = loocv_results['decision_tree_scores']

svm_scores_holdout = holdout_results['svm_score']
svm_scores_kfold = kfold_results['svm_scores']
svm_scores_loocv = loocv_results['svm_scores']

# Calculate ROC and other metrics for Decision Tree for different methods
fpr_dt_holdout, tpr_dt_holdout, _ = roc_curve(
    y_val, decision_tree.predict_proba(X_val)[:, 1])
roc_auc_dt_holdout = roc_auc_score(y_val, decision_tree.predict(X_val))

fpr_dt_kfold, tpr_dt_kfold, _ = roc_curve(
    y, decision_tree.predict_proba(X)[:, 1])
roc_auc_dt_kfold = roc_auc_score(y, decision_tree.predict(X))

fpr_dt_loocv, tpr_dt_loocv, _ = roc_curve(
    y, decision_tree.predict_proba(X)[:, 1])
roc_auc_dt_loocv = roc_auc_score(y, decision_tree.predict(X))

# Calculate ROC and other metrics for SVM for different methods
fpr_svm_holdout, tpr_svm_holdout, _ = roc_curve(
    y_val, svm.decision_function(X_val))
roc_auc_svm_holdout = roc_auc_score(y_val, svm.predict(X_val))

fpr_svm_kfold, tpr_svm_kfold, _ = roc_curve(y, svm.decision_function(X))
roc_auc_svm_kfold = roc_auc_score(y, svm.predict(X))

fpr_svm_loocv, tpr_svm_loocv, _ = roc_curve(y, svm.decision_function(X))
roc_auc_svm_loocv = roc_auc_score(y, svm.predict(X))

# Plot ROC curve for Decision Tree with different methods
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(fpr_dt_holdout, tpr_dt_holdout, color='blue', lw=2,
         label='ROC curve (Holdout) (area = %0.2f)' % roc_auc_dt_holdout)
plt.plot(fpr_dt_kfold, tpr_dt_kfold, color='green', lw=2,
         label='ROC curve (k-fold CV) (area = %0.2f)' % roc_auc_dt_kfold)
plt.plot(fpr_dt_loocv, tpr_dt_loocv, color='red', lw=2,
         label='ROC curve (LOOCV) (area = %0.2f)' % roc_auc_dt_loocv)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree (Multiple Methods)')
plt.legend(loc='lower right')

# Plot ROC curve for SVM with different methods
plt.subplot(1, 3, 2)
plt.plot(fpr_svm_holdout, tpr_svm_holdout, color='blue', lw=2,
         label='ROC curve (Holdout) (area = %0.2f)' % roc_auc_svm_holdout)
plt.plot(fpr_svm_kfold, tpr_svm_kfold, color='green', lw=2,
         label='ROC curve (k-fold CV) (area = %0.2f)' % roc_auc_svm_kfold)
plt.plot(fpr_svm_loocv, tpr_svm_loocv, color='red', lw=2,
         label='ROC curve (LOOCV) (area = %0.2f)' % roc_auc_svm_loocv)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM (Multiple Methods)')
plt.legend(loc='lower right')

# Additional plot
plt.subplot(1, 3, 3)
plt.plot(fpr_dt_holdout, tpr_dt_holdout, color='blue', lw=2,
         label='ROC curve (Holdout) (area = %0.2f)' % roc_auc_dt_holdout)
plt.plot(fpr_dt_kfold, tpr_dt_kfold, color='green', lw=2,
         label='ROC curve (k-fold CV) (area = %0.2f)' % roc_auc_dt_kfold)
plt.plot(fpr_dt_loocv, tpr_dt_loocv, color='red', lw=2,
         label='ROC curve (LOOCV) (area = %0.2f)' % roc_auc_dt_loocv)

# Plot a single ROC curve for SVM with different methods
plt.plot(fpr_svm_holdout, tpr_svm_holdout, color='cyan', lw=2,
         label='ROC curve (SVM Holdout) (area = %0.2f)' % roc_auc_svm_holdout)
plt.plot(fpr_svm_kfold, tpr_svm_kfold, color='magenta', lw=2,
         label='ROC curve (SVM k-fold CV) (area = %0.2f)' % roc_auc_svm_kfold)
plt.plot(fpr_svm_loocv, tpr_svm_loocv, color='orange', lw=2,
         label='ROC curve (SVM LOOCV) (area = %0.2f)' % roc_auc_svm_loocv)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree and SVM (Multiple Methods)')
plt.legend(loc='lower right')
plt.show()