# credit-risk-classification
# Module 20 Challenge - Supervised Machine Learning

The objective of this challenge was to use various machine learning techniques to train and evaluate a model based on loan risk.

# Overview of the Analysis
The data set used "lending_data.csv" provided historical lending activity from a lending services company. The
"loan_status" column provided the classification for a "healthy loan" with a value of 0 and a "high-risk loan"
with a value of 1. The models built from this data set would help identify the creditworthiness of borrowers.

In general, the process to train and evaluate the different models followed these steps:
1- Read data source and save to a pandas data frame.
2- Separate the data into the "target" (y) and "features" (X).
3- Encode data if needed (ie. get dummies function).
4- Separate data into "Training" and "Test" data sets.
5- Scale the data as needed.
6- Instantiate the model (ie. Logistic Regression).
7- Fit the model.
8- Make predictions using the "Test" data set.
9- Generate a classification report to evaluate the accuracy of the model.

Following the starter Jupyter notebook (credit_risk_classification.ipynb), the Logistic Regression model was 
applied to the data set. In addition, the Random Forest and SVM models were also evaluated on the data set.

# Results
Machine Learning Model 1 : Logistic Regression
Accuracy Score : 0.9918489475856377
precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384

Machine Learning Model 2 : Random Forest
Accuracy Score : 0.9917457697069748
Classification Report
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.90      0.87       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384

Machine Learning Model 3 : SVM
Accuracy Score: 0.994
                precision    recall  f1-score   support

  healthy_loan       1.00      0.99      1.00     18765
high_risk_loan       0.84      0.98      0.91       619

      accuracy                           0.99     19384
     macro avg       0.92      0.99      0.95     19384
  weighted avg       0.99      0.99      0.99     19384

# Summary
All three models show very similar results in terms of accuracy score, precision, recall and f1-score.
The Random Forest model is the least accurate of the three models.
All three models are equally good at identifying the healthy loans (0) with a precision of 1.0, however
the precision to identify the high-risk loans (1) is not as good at a level of 0.85.

I would recommend the Logistic Regression or the SVM models for this classification task. However, the performance of the model
will depend on the importance of the prediction of healthy loans vs high-risk loans. So if it is important to
ensure the prediction of high-risk loans over healthy loans, both these models will not be highly accurate. 
