# Fraud Detection Model

This project uses a logistic regression model to detect fraudulent transactions in credit card data. The original dataset has a heavily imbalanced class distribution, with a large number of normal transactions and a smaller number of fraudulent transactions. The data was undersampled to balance the class distribution and create a new dataset for analysis. The logistic regression model was trained on the undersampled data and evaluated on a held-out test set. The evaluation metrics used were accuracy, precision, recall, F1 score, confusion matrix, and ROC curve.

## Requirements


## Usage
The code can be run using Python and the packages listed in the requirements. Simply download the `creditcard.csv` dataset and run the `fraud_detection.py` script. The script will perform the following steps:
1. Clone the repository: git clone https://github.com/Hotmansifu/credit-card-fraud-detection.git
Navigate to the directory: ```cd credit-card-fraud-detection```
Run the script: ```python main.py```
- Read in the credit card data
- Balance the class distribution by undersampling the normal transactions
- Split the data into training and test sets
- Train a logistic regression model on the training data
- Evaluate the model on the test data and calculate various performance metrics
- Plot the ROC curve

## Dataset
The dataset used in this code is the Credit Card Fraud Detection dataset from Kaggle. The data has been undersampled to have a similar distribution of normal and fraudulent transactions.


## Results
The logistic regression model achieved an accuracy of **XXX** on the test data, with a precision of **XXX**, recall of **XXX**, and F1 score of **XXX**. The confusion matrix and ROC curve are also plotted to provide additional insights into the model's performance.

## Note
This project is intended for educational purposes and should not be used for actual fraud detection in production. Further data preprocessing, feature engineering, and model tuning may be necessary for real-world applications.
