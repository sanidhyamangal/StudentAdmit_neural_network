import numpy as np # for matrix maths 
import pandas as pd # for data frame 

admissions = pd.read_csv('./student_data.csv')

# Make dummy variable for rank column 
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
# drop rank column 
data = data.drop('rank', axis=1)

# Standarize the features 
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data[field] = (data[field] - mean) / std

# Split data into test set and train set (90-10)
sample = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets 
features, target = data.drop('admit', axis=1), data['admit']
features_test, target_test = test_data.drop('admit', axis=1), test_data['admit']