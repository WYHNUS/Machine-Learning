import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('Selected_Column_Data.csv')
# now df has no missing values
df = df.dropna(0, 'any')
df.apply(pd.to_numeric)

# one hot encoding
df = pd.get_dummies(df, columns = ['race_o', 'race'])

# random select 60% for training, 40% for testing
train, test = train_test_split(df, test_size = 0.4)

y_train_with_id = train.loc[:, ['iid', 'match']]
X_train_with_id = train.drop('match', axis=1)
y_train = y_train_with_id.loc[:, ['match']]
X_train = X_train_with_id.drop('iid', axis=1)

y_test_with_id = test.loc[:, ['iid', 'match']]
X_test_with_id = test.drop('match', axis=1)
y_test = y_test_with_id.loc[:, ['match']]
X_test = X_test_with_id.drop('iid', axis=1)

y_train.to_csv('y_train.csv', index=False, encoding='utf-8')
X_train.to_csv('X_train.csv', index=False, encoding='utf-8')
y_test.to_csv('y_test.csv', index=False, encoding='utf-8')
X_test.to_csv('X_test.csv', index=False, encoding='utf-8')