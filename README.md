# CREDIT RISK CLASSIFICATION
This project make me a champion of The Data Analytics Competition

![image](https://github.com/user-attachments/assets/2c72ac22-cb00-4cc3-a32a-089906db1dc2)


# Introduce
This project involves building a machine learning model to classify credit scores. The notebook includes data preprocessing, model training, and evaluation steps.

# Table of Contents
- Data Loading and Exploration
- Data Preprocessing
- Feature Engineering
- Model Training and Evaluation
- Model Saving


### Data Loading and Exploration
```
import pandas as pd
import numpy as np

train = pd.read_csv(r"D:\\CHUNG KẾT DA final\\dataset\\Train_DA.csv")
test = pd.read_csv(r"D:\\CHUNG KẾT DA final\\dataset\\Test_DA.csv")

train.head()
```

This block imports necessary libraries (pandas and numpy) and loads the training and test datasets from the specified file paths. The train.head() function displays the first few rows of the training dataset for an initial exploration.

### Data Preprocessing

```
train.isnull().sum()
test.isnull().sum()
```
This block checks for missing values in the training and test datasets by summing up the null values in each column.

```
train = train.fillna(train.median())
test = test.fillna(test.median())
```

This block fills any missing values in the training and test datasets with the median value of the respective columns.

### Feature Engineering

```
train['Feature1'] = train['Feature1'].apply(lambda x: np.log1p(x))
test['Feature1'] = test['Feature1'].apply(lambda x: np.log1p(x))
```
This block applies a log transformation to Feature1 in both the training and test datasets to handle skewness in the data.

### Model Training and Evaluation

```
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X = train.drop(columns=['Target'])
y = train['Target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")
```
Explanation:
- This block imports necessary libraries for model training and evaluation (train_test_split from sklearn.model_selection, XGBClassifier from xgboost, and accuracy_score from sklearn.metrics).
- Splits the training data into training and validation sets.
- Initializes an XGBClassifier, fits it to the training data, and predicts on the validation set.
- Calculates and prints the accuracy of the model on the validation set.

### Model Saving
```
submission_file = test[['Case_ID']].copy()
submission_file['Approved_Flag'] = xgb.predict(test.drop(columns=['Case_ID']))
submission_file.head()
```
This block creates a submission file by predicting the target variable (Approved_Flag) for the test dataset using the trained model and including the Case_ID for each record.

# Conclusion

Conclusion

In this project, we successfully developed a machine learning model to classify credit scores. The key steps involved in the process were:

- Data Loading and Exploration: We loaded the training and test datasets, and conducted an initial exploration to understand the data structure and identify any missing values.

- Data Preprocessing: We handled missing values by filling them with the median values of the respective columns, ensuring our datasets were clean and ready for analysis.

- Feature Engineering: We applied a log transformation to Feature1 to reduce skewness, which helps in improving the performance of the machine learning model.

- Model Training and Evaluation: We split the training data into training and validation sets and trained an XGBoost classifier. We evaluated the model using accuracy as the metric and achieved satisfactory results on the validation set.

- Model Saving: We generated predictions for the test dataset, saved these predictions to an Excel file, and also saved the trained model using pickle for future use.

The project demonstrates the importance of a structured approach in machine learning, from data preprocessing to model evaluation. By following these steps, we ensure that the model is robust, reliable, and ready for deployment. Future work can include exploring other machine learning algorithms, tuning hyperparameters to further improve the model's performance, and applying the model to different datasets.

This project provides a solid foundation for anyone looking to understand the end-to-end process of building and deploying a machine learning model for classification tasks.
