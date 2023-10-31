from sklearn.model_selection import GridSearchCV

# Define hyperparameter grids for Decision Tree and SVM
decision_tree_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}

# Create GridSearchCV objects for both models
decision_tree_grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=decision_tree_param_grid,
    scoring='accuracy',  # Choose an appropriate scoring metric
    cv=5,
    n_jobs=-1
    # Cross-validation folds
)

svm_grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=svm_param_grid,
    scoring='accuracy',  # Choose an appropriate scoring metric
    cv=5,
    n_jobs=-1
)

# Perform grid search for Decision Tree
decision_tree_grid_search.fit(X_train, y_train)

# Perform grid search for SVM
svm_grid_search.fit(X_train, y_train)

# Print the best hyperparameters and their scores
print("Best Decision Tree Hyperparameters: ",
      decision_tree_grid_search.best_params_)
print("Best Decision Tree Score: ", decision_tree_grid_search.best_score_)

print("Best SVM Hyperparameters: ", svm_grid_search.best_params_)
print("Best SVM Score: ", svm_grid_search.best_score_)

# Evaluate the best models on the validation set
best_decision_tree = decision_tree_grid_search.best_estimator_
best_svm = svm_grid_search.best_estimator_

decision_tree_score = best_decision_tree.score(X_val, y_val)
svm_score = best_svm.score(X_val, y_val)

print(
    f"Accuracy of Best Decision Tree Model on Validation Set: {decision_tree_score:.2f}")
print(f"Accuracy of Best SVM Model on Validation Set: {svm_score:.2f}")