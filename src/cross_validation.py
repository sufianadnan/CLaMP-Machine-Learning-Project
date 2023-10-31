import logging
import pickle

# Configure the logging settings
log_file = "error_log.txt"
logging.basicConfig(filename=log_file, level=logging.ERROR,
                    format="%(asctime)s [%(levelname)s] - %(message)s")

# Create instances of your machine learning models (e.g., Decision Tree and SVM)
decision_tree = DecisionTreeClassifier()
svm = SVC()

# 1. Holdout Method
try:
    # Split the data into a training set and a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    decision_tree.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    decision_tree_score = decision_tree.score(X_val, y_val)
    svm_score = svm.score(X_val, y_val)

    print(
        f"Holdout Method - Decision Tree (J48) Accuracy: {decision_tree_score:.2f}")
    print(f"Holdout Method - SVM Accuracy: {svm_score:.2f}")

    # Save results of Holdout Method
    with open('pkl/holdout_results.pkl', 'wb') as file:
        pickle.dump({
            'decision_tree_score': decision_tree_score,
            'svm_score': svm_score
        }, file)
except Exception as e:
    logging.error(f"Holdout Method error: {str(e)}")

# 2. k-fold Cross-Validation
try:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    decision_tree_scores = cross_val_score(decision_tree, X, y, cv=kf)
    svm_scores = cross_val_score(svm, X, y, cv=kf)

    print("k-fold Cross-Validation - Decision Tree (J48) Scores:",
          decision_tree_scores)
    print("k-fold Cross-Validation - SVM Scores:", svm_scores)
    print(
        f"Average Decision Tree (J48) Accuracy: {decision_tree_scores.mean():.2f}")
    print(f"Average SVM Accuracy: {svm_scores.mean():.2f}")

    # Save results of K-fold Cross-Validation
    with open('pkl/kfold_results.pkl', 'wb') as file:
        pickle.dump({
            'decision_tree_scores': decision_tree_scores,
            'svm_scores': svm_scores
        }, file)
except Exception as e:
    logging.error(f"k-fold Cross-Validation error: {str(e)}")

# 3. Leave-One-Out Cross-Validation (LOOCV)
try:
    loo = LeaveOneOut()
    decision_tree_scores = cross_val_score(
        decision_tree, X, y, cv=loo, n_jobs=-1)
    svm_scores = cross_val_score(svm, X, y, cv=loo, n_jobs=-1)

    print("Leave-One-Out Cross-Validation - Decision Tree (J48) Scores:",
          decision_tree_scores)
    print("Leave-One-Out Cross-Validation - SVM Scores:", svm_scores)
    print(
        f"Decision Tree (J48) Accuracy with LOOCV: {decision_tree_scores.mean():.2f}")
    print(f"SVM Accuracy with LOOCV: {svm_scores.mean():.2f}")

    # Save results of Leave-One-Out Cross-Validation
    with open('pkl/loocv_results.pkl', 'wb') as file:
        pickle.dump({
            'decision_tree_scores': decision_tree_scores,
            'svm_scores': svm_scores
        }, file)
except Exception as e:
    logging.error(f"LOOCV error: {str(e)}")