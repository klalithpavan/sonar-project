import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Get the data

data = pd.read_csv("sonar.data" , header = None)

data.groupby(60).mean()

# Training and Testing the data

x = data.drop(columns = 60,axis=1)
y = data[60]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify =y , random_state= 1 )

# Model training by Logistic Regression

lr = LogisticRegression()
lr.fit(x_train,y_train)
# predictions
prediction_x_train = lr.predict(x_train)
accuracy = accuracy_score(prediction_x_train,y_train)

prediction_x_test = lr.predict(x_test)
accuracy_x_test = accuracy_score(prediction_x_test,y_test)

# example sample

input = (0.0519,0.0548,0.0842,0.0319,0.1158,0.0922,0.1027,0.0613,0.1465,0.2838,0.2802,0.3086,0.2657,0.3801,0.5626,0.4376,0.2617,0.1199,0.6676,0.9402,0.7832,0.5352,0.6809,0.9174,0.7613,0.8220,0.8872,0.6091,0.2967,0.1103,0.1318,0.0624,0.0990,0.4006,0.3666,0.1050,0.1915,0.3930,0.4288,0.2544,0.1152,0.2196,0.1879,0.1437,0.2147,0.2360,0.1125,0.0254,0.0285,0.0178,0.0052,0.0081,0.0120,0.0045,0.0121,0.0097,0.0085,0.0047,0.0048,0.0053)

#input data into array
input_array = np.asarray(input)
input_reshape = input_array.reshape(1,-1)

prediction = lr.predict(input_reshape)
print(prediction)
if prediction[0]=="M":
  print("ALERT! its a mine ")

else:
  print("its a rock")



