# Drop the specified columns from the DataFrame
df = df.drop(['e_cblp', 'e_cp', 'e_cparhdr', 'e_maxalloc', 'e_sp',
             'e_lfanew', 'NumberOfSections', 'CreationYear'], axis=1)

# Fill any missing values in the DataFrame with zero and update df in place.
df.fillna(0, inplace=True)