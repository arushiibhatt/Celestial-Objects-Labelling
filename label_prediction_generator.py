import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('celestial_train.csv')
test = pd.read_csv('celestial_test.csv')

cols = ['id', 'alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift', 'plate', 'MJD', 'fiber_ID']
X = df[cols]
Y = df['class']

clf = RandomForestClassifier()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9)
clf.fit(X_train, Y_train)

clf.predict(X_test)
print(clf.score(X_test, Y_test))

values = list(clf.predict(test))

with open('celestial_submission.csv', mode='w', newline='') as file: 
    writer = csv.writer(file) 
    writer.writerow(['id', 'output']) 
    for i in range(len(values)): 
        new_value = values[i]  
        writer.writerow([str(50000+i), values[i]])
