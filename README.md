# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)

features = ['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target = 'class'
x = data[features]
y = data[target]

x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

svm = SVC()

param_grid = {
    'C': [0.1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale','auto']
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train,y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


accuracy = accuracy_score(y_test, y_pred)
print("Name:Ponsriram P")
print("Register Number:212225240105")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="840" height="705" alt="image" src="https://github.com/user-attachments/assets/18c81f8e-dd68-4b5a-8d6a-fd297b17466b" />
<img width="686" height="302" alt="image" src="https://github.com/user-attachments/assets/2151ad08-f09a-419e-9bdd-5d5f4ae19e37" />
<img width="752" height="593" alt="image" src="https://github.com/user-attachments/assets/30780a4d-1732-4530-b693-a6e269a25b92" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
