# Get unique values in the 'packer_type' column and create a mapping to integers
pt = df['packer_type'].unique()
p_types = {pt[i]: i for i in range(len(pt))}
temp = []
for t in df['packer_type']:
    temp.append(p_types[t])
df['pt_num'] = temp
cl = df.pop('class')
df.pop('packer_type')

# 'packer_type' column changed to 'pt_num' column with corresponding integers

# Lists to store the accuracy scores for Decision Tree and SVM
decision_tree_scores = []
svm_scores = []

# Lists to store split size and random state for each iteration
split_sizes_used = []
random_states_used = []

# Specify the split sizes and random states
split_sizes = [0.2, 0.3]
random_states = [0, 42]

for split_size in split_sizes:
    for random_state in random_states:
        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            df, cl, test_size=split_size, random_state=random_state)

        # Train a Decision Tree (J48) model
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)

        # Train a Support Vector Machine (SVM) model
        svm = SVC()
        svm.fit(X_train, Y_train)

        # Evaluate the models
        decision_tree_score = decision_tree.score(X_test, Y_test)
        svm_score = svm.score(X_test, Y_test)

        # Store the accuracy scores in their respective lists
        decision_tree_scores.append(decision_tree_score)
        svm_scores.append(svm_score)

        # Store the split size and random state
        split_sizes_used.append(split_size)
        random_states_used.append(random_state)

        # Print the results for this iteration
        print(f"Split Size: {split_size}, Random State: {random_state}")
        print(f"Decision Tree (J48) Accuracy: {decision_tree_score:.2f}")
        print(f"SVM Accuracy: {svm_score:.2f}")
        print("===================================")

# Create bar charts for accuracy
x_labels = [f"Split Size: {split_size}, Random State: {random_state}" for split_size,
            random_state in zip(split_sizes_used, random_states_used)]
x = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35

# Create bar charts for Decision Tree and SVM accuracy
bar1 = ax.bar(x - width/2, decision_tree_scores,
              width, label='Decision Tree (J48)')
bar2 = ax.bar(x + width/2, svm_scores, width, label='SVM')

ax.set_xlabel('Split Size, Random State')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy for Different Split Sizes and Random States')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45)
ax.legend()

# Add the accuracy values inside the bars
for bar, score in zip(bar1, decision_tree_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() -
            0.05, f'{score:.2f}', ha='center', color='white')

for bar, score in zip(bar2, svm_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() -
            0.05, f'{score:.2f}', ha='center', color='black')

# Display the bar chart
plt.tight_layout()
plt.show()