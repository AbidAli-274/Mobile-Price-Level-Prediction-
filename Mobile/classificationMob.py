import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from scikitplot.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

# Read the data
data = pd.read_csv("Mobile.csv")
Y = data['price_range']
X = data.drop(['price_range'], axis=1)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Perform feature scaling
std_scale = StandardScaler()
X_train_scaled = std_scale.fit_transform(X_train)
X_test_scaled = std_scale.transform(X_test)

# Initialize and train the classifier
clf = DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=0)
clf.fit(X_train_scaled, Y_train)

# Make predictions on the training set
train_pred = clf.predict(X_train_scaled)
print('Training Confusion Matrix:')
print(confusion_matrix(Y_train, train_pred, labels=[0, 1, 2, 3]))
print('Training Score:', clf.score(X_train_scaled, Y_train))

# Make predictions on the testing set
test_pred = clf.predict(X_test_scaled)
print('Testing Confusion Matrix:')
print(confusion_matrix(Y_test, test_pred, labels=[0, 1, 2, 3]))
print('Testing Score:', clf.score(X_test_scaled, Y_test))

# Plot ROC curve for testing set
probs = clf.predict_proba(X_test_scaled)
skplt.metrics.plot_roc_curve(Y_test, probs)
plt.show()

# Save the trained model and StandardScaler
pickle.dump(clf, open('./Mobilemodel.pkl', 'wb'))
pickle.dump(std_scale, open('./StandardScaler.pkl', 'wb'))

# Make a prediction on a new data point
#new_data = [[5000,1,142,1,150,1,1128,0.36,96,8,16,1381,1311,16,5.2,5.2,20,1,1,1]]
#new_data_scaled = std_scale.transform(new_data)
#predicted_data = clf.predict(new_data_scaled)
#print("Predicted data:", predicted_data)

print('Score:', clf.score(X_train_scaled, Y_train))
#print('Score:', clf.score(X_test_scaled, Y_test))
