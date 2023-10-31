# Assuming you have imported your DataFrame as 'df'
df['class'].value_counts()
# Create a figure for the plot with the specified size.
fig = plt.figure(figsize=(5, 4))
# Plot the counts of each class in the 'class' column and display it as a bar chart.
ax = df['class'].value_counts().plot(kind='bar')
# Add labels to the plot to indicate the classes (1=malware, 0=benign).
plt.xticks([0, 1], ['0=benign', '1=malware'])
# Add values on top of each bar
for i, v in enumerate(df['class'].value_counts()):
    ax.text(i, v, str(v), ha='center', va='bottom')
# Show the plot.
plt.show()