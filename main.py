import pandas as pd
penguins = pd.read_csv('penguins_cleaned.csv')
# print(penguins)

# Encoding categorical/ nominal/ ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
# print(df)
target = 'species'
encode = ['sex','island']

# Here, we get each ordinal feature/ variable in the 'col' list and encode it to 1 and 0. Then, we add it, after encoding to
# the dataframe and delete the original column
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    # print(df[col].head())
    df = pd.concat([df,dummy], axis=1)
    del df[col] # delete the original column

# Here, we encode the target 'species' in a dictionary then we create a custome function that returns the encoding by entering the species name
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

# print(target_encode('Adelie'))

# Here, we apply the custom function 'target_encode' to apply the encoding in the dataframe
df['species'] = df['species'].apply(target_encode)

# Separating X (the predictors) and y (the target)
X = df.drop('species', axis=1)
Y = df['species']

# Build random forest classification model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model so we don't need to build it everytime we change the input parameters (This will save the resources)
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))