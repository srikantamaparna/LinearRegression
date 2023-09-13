#importing packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#read house dataset and check for null data
data=pd.read_csv("D:\id3\kc_house_data.csv\kc_house_data.csv") #21614 King County, Washington State, USA
l1=[]
for i in data.date:
    l1.append(i[:4])
conv_dates=[1 if values=='2014' else 0 for values in l1]
data['date']=conv_dates
# convert any string data to numerical data using suitable conversion
#identify dependent and independent variables
y=data[['price']].values #dependent
x=data.drop(['id','price'],axis=1).values
#split data into train data and test data
#print(x[0])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#define and train(fit) linear regression model using training data
l=LinearRegression()
l.fit(x_train,y_train)
y_predict=l.predict(x_test)
f=np.array([[0,3,2,1870,5000,2,0,0,3,8,0,1955,0,98143,47.37,-122.76,1345,5000]])
p=l.predict(f)
print('Coefficient of determination : %.3f'%r2_score(y_test,y_predict))
print("predicted house price is :",p)



