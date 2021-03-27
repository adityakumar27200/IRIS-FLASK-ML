# import library and load dataset
import numpy as np
import pandas as pd
df=pd.read_csv('iris.data')

# seperate data values and target values
x=np.array(df.iloc[:,0:4])
y=np.array(df.iloc[:,4:])

# Encode the  target values into numeric form
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

# split data into test-set and train-set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#load the Support Vector Classification model
from sklearn.svm import SVC
svc=SVC(kernel='linear').fit(x_train,y_train)

#predict the model
model=svc.predict(x_test)
#print(model)
#print(y_test)


# Evaluate the confusion matrix, classification report and accuracy score

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test,model)
print("-----------Confusion Matrix--------------")
print(result)
result1 = classification_report(y_test,model)
print("-----------Classification Report--------------")
print (result1)
result2 = accuracy_score(y_test,model)
print("Accuracy---",result2)


#save the model into using pickle
import pickle
pickle.dump(svc,open('iriss.pkl','wb'))
