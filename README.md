# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Collection and Preprocessing Load the placement dataset and remove unnecessary columns. Check for missing and duplicate values, and convert all categorical variables into numerical form using Label Encoding.

Step 2: Feature Selection and Data Splitting Separate the dataset into independent variables (features) and the dependent variable (placement status). Split the data into training and testing sets.

Step 3: Model Training Apply the Logistic Regression algorithm on the training data to build the prediction model.

Step 4: Prediction and Evaluation Use the trained model to predict placement status on test data and evaluate the performance using accuracy score, confusion matrix, and classification report.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Ypogesh G 
RegisterNumber:  25009804
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:/Users/91908/Downloads/Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])
datal

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
y_pred

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)


classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



```

## Output:
HEAD
<img width="485" height="43" alt="Screenshot 2026-02-04 192158" src="https://github.com/user-attachments/assets/0686f60f-ce8a-435f-a815-897105ed2676" />
<img width="949" height="184" alt="Screenshot 2026-02-04 192209" src="https://github.com/user-attachments/assets/31db91d7-3c5d-402a-9ff6-b9e7c4c806a2" />

COPY
<img width="410" height="47" alt="Screenshot 2026-02-04 192318" src="https://github.com/user-attachments/assets/30705170-b28d-41e2-8715-1c4f56b26ddb" />
<img width="825" height="165" alt="Screenshot 2026-02-04 192326" src="https://github.com/user-attachments/assets/a2f92597-08fa-4e9f-a253-fe9e39a63a85" />

FIT TRANSFORM

<img width="479" height="178" alt="Screenshot 2026-02-04 192432" src="https://github.com/user-attachments/assets/13ac366d-4031-410d-9f96-5b9018c0bb52" />
<img width="785" height="379" alt="Screenshot 2026-02-04 192443" src="https://github.com/user-attachments/assets/2e4d23a1-693e-4f02-9749-63d6e8ddac72" />
<img width="699" height="132" alt="Screenshot 2026-02-04 192456" src="https://github.com/user-attachments/assets/1dce5daf-820a-4eb5-9bba-b2e3abe1730a" />

LOGISTIC REGRESSION
ACCURACY SCORE
<img width="320" height="66" alt="Screenshot 2026-02-04 192702" src="https://github.com/user-attachments/assets/ca8cabdf-a430-47ac-b622-9e25fab18357" />

CONFUSION MATRIX
<img width="331" height="94" alt="Screenshot 2026-02-04 192709" src="https://github.com/user-attachments/assets/03b4d338-fb1f-44dd-8b97-954c06f15926" />

CLASSIFICATION REPORT & PREDICTION
<img width="720" height="268" alt="Screenshot 2026-02-04 192716" src="https://github.com/user-attachments/assets/8a350fce-cad9-458e-92cf-d4e675e9815c" />
<img width="753" height="210" alt="Screenshot 2026-02-04 192724" src="https://github.com/user-attachments/assets/fe4e2d51-00c2-4304-911a-b0aedbc4296d" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
