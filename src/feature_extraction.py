
# Select only the numeric columns from the DataFrame
numeric_data = df.select_dtypes(include=[np.number])

# Separate the features (X) and the target variable (y).
X = numeric_data.drop(columns=['class'])
y = df['class']

# Feature selection using chi-squared (chi2) score function
# Create a SelectKBest instance with the chi-squared score function and select the top k features.
chi2_selector = SelectKBest(score_func=chi2, k=10)
X_chi2 = chi2_selector.fit_transform(X, y)

# Get the indices of the selected features.
selected_feature_indices_chi2 = chi2_selector.get_support(indices=True)

# Get the names of the selected features.
selected_feature_names_chi2 = X.columns[selected_feature_indices_chi2]

# Print the names of the selected features for chi-squared.
print("Selected Features (Chi-squared):")
print(selected_feature_names_chi2)

# Feature selection using mutual information score function
# Create a SelectKBest instance with the mutual information score function and select the top k features.
mutual_info_selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_mutual_info = mutual_info_selector.fit_transform(X, y)

# Get the indices of the selected features.
selected_feature_indices_mutual_info = mutual_info_selector.get_support(
    indices=True)

# Get the names of the selected features.
selected_feature_names_mutual_info = X.columns[selected_feature_indices_mutual_info]

# Print the names of the selected features for mutual information.
print("\nSelected Features (Mutual Information):")
print(selected_feature_names_mutual_info)