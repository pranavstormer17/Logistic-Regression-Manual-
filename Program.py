import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
import seaborn as sn 
import matplotlib.pyplot as plt 

col_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

data = pd.read_csv('/home/pranavstormer17/Downloads/diabetes.csv', header=0, names=col_names)

print("Data shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())
print("\nNull values:\n", data.isnull().sum()) 

feature_cols = ['Pregnancies','Insulin','BMI','Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
x = data[feature_cols]

y = data.Outcome 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)
print("\nData splits (x_train, y_train, x_test, y_test):")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

conf_mat = metrics.confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:\n', conf_mat)

Accuracy_score = metrics.accuracy_score(y_test, y_pred)
print('Accuracy Score: ', Accuracy_score)
print('Accuracy in Percentage: ', int(Accuracy_score*100), '%')

conf_mat_df = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(conf_mat_df, annot=True)

plt.show()
