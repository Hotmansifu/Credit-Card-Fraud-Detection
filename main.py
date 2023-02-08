import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

credit_card_data = pd.read_csv("creditcard.csv")  

# Separate the data for analysis
legit = credit_card_data[credit_card_data.Class== 0]
fraud = credit_card_data[credit_card_data.Class== 1]
print("Legitimate transactions:", legit.shape)
print("Fraudulent transactions:", fraud.shape)

# Undersample the data to create a new dataset with a similar distribution of normal and fraudulent transactions
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis = 0)
print("Class distribution in the new dataset:")
print(new_dataset["Class"].value_counts())

# Split the new dataset into features (x) and target (y)
x = new_dataset.drop(columns="Class", axis=1)
y = new_dataset["Class"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Fit a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Evaluate the accuracy of the model on the training data
y_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train_prediction, y_train)
print("Accuracy on training data:", training_data_accuracy)

# Evaluate the accuracy of the model on the test data
y_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test_prediction, y_test)
print("Accuracy on test data:", test_data_accuracy)

confusion_matrix = confusion_matrix(y_test, y_test_prediction)
print("Confusion Matrix : \n", confusion_matrix)

precision = precision_score(y_test, y_test_prediction)
recall = recall_score(y_test, y_test_prediction)
f1_score = f1_score(y_test, y_test_prediction)

print("Precision : ", precision)
print("Recall : ", recall)
print("f1_score : ", f1_score)

# Plotting the ROC curve
from sklearn.metrics import roc_auc_score, roc_curve

auc = roc_auc_score(y_test, y_test_prediction)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prediction)

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
