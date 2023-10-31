import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# Load the k-fold results from the pickle file
with open('pkl/kfold_results.pkl', 'rb') as file:
    results = pickle.load(file)
    decision_tree_scores = results['decision_tree_scores']
    svm_scores = results['svm_scores']

# Define function to plot ROC curve


def plot_roc_curve(fpr, tpr, roc_auc, title, label):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label=f'{label} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc='lower right')
    plt.show()


# Loop through each fold and generate metrics
for fold, (dt_score, svm_score) in enumerate(zip(decision_tree_scores, svm_scores)):
    print(f"Fold {fold+1} - Decision Tree (J48) Score: {dt_score:.2f}")
    print(f"Fold {fold+1} - SVM Score: {svm_score:.2f}")

    # Generate confusion matrix for Decision Tree
    dt_pred = decision_tree.predict(X)
    dt_cm = confusion_matrix(y, dt_pred)
    plt.figure()
    sns.heatmap(dt_cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - Decision Tree (Fold {fold+1})")
    plt.show()

    # Generate classification report for Decision Tree
    dt_report = classification_report(y, dt_pred)
    print(
        f"Classification Report - Decision Tree (Fold {fold+1}):\n{dt_report}")

    # Generate ROC curve for Decision Tree
    dt_fpr, dt_tpr, _ = roc_curve(y, dt_pred)
    dt_roc_auc = auc(dt_fpr, dt_tpr)
    plot_roc_curve(dt_fpr, dt_tpr, dt_roc_auc,
                   "Decision Tree", f"Fold {fold+1}")

    # Generate confusion matrix for SVM
    svm_pred = svm.predict(X)
    svm_cm = confusion_matrix(y, svm_pred)
    plt.figure()
    sns.heatmap(svm_cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - SVM (Fold {fold+1})")
    plt.show()

    # Generate classification report for SVM
    svm_report = classification_report(y, svm_pred)
    print(f"Classification Report - SVM (Fold {fold+1}):\n{svm_report}")

    # Generate ROC curve for SVM
    svm_fpr, svm_tpr, _ = roc_curve(y, svm_pred)
    svm_roc_auc = auc(svm_fpr, svm_tpr)
    plot_roc_curve(svm_fpr, svm_tpr, svm_roc_auc, "SVM", f"Fold {fold+1}")