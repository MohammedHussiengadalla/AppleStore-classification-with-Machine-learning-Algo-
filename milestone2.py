import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import set_printoptions
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import  pickle


def remove_nulls(data_file):
    data_file = data_file.drop(data_file[data_file["id"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["track_name"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["size_bytes"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["currency"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["price"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["rating_count_tot"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["rating_count_ver"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["vpp_lic"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["ver"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["cont_rating"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["prime_genre"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["sup_devices.num"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["ipadSc_urls.num"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["lang.num"].isnull()].index, axis=0)
    data_file = data_file.drop(data_file[data_file["rate"].isnull()].index, axis=0)
    return data_file



data = pd.read_csv('AppleStore_training_classification.csv')

categories = list(data["prime_genre"].unique())
# delete empty cells ,invalid data
data = remove_nulls(data)
#for string prime genre
categories.remove("0")
a = data.loc[data["prime_genre"] == "0"]
data = data.drop(int(a.index.values), axis=0)
data=data.drop(columns=['id','track_name','currency','ver'])

levels = list(data["rate"].unique())

data['cont_rating'] = data['cont_rating'].str.replace('+', '').astype(int)

data = pd.concat([ pd.get_dummies(data.prime_genre),data.drop('prime_genre', 1)], 1)

data['rating_count_tot'] = (data['rating_count_tot'] - min(data['rating_count_tot'])) / (
    max(data['rating_count_tot'] - min(data['rating_count_tot'])))
data['size_bytes'] = (data['size_bytes'] - min(data['size_bytes'])) / (
    max(data['size_bytes'] - min(data['size_bytes'])))

###################################################################################




data['rate']= data['rate'].str.replace('High', '2')
data['rate']= data['rate'].str.replace('Low', '0')
data['rate']= data['rate'].str.replace('Intermediate', '1')
data['rate']=data['rate'].astype(int)
X=data.iloc[:,:-1]
Y=data['rate']
fs = SelectKBest(score_func=f_classif, k=4)
X_selected = fs.fit_transform(X, Y)
X_train, X_test ,Y_train,Y_test=train_test_split(X_selected ,Y,test_size=0.2,random_state=0)


model1=SVC(C=10)
model1.fit(X_train,Y_train)
ypredcate =model1.predict(X_test)
file_name ='svm_weight.sav'
pickle.dump(model1,open(file_name,'wb'))
#####################################################################


model2=LogisticRegression(C=200)
model2.fit(X_train,Y_train)
ypredcate =model2.predict(X_test)

file_name2 ='logistic_weight.sav'
pickle.dump(model2,open(file_name2,'wb'))

######################################################################

model3=DecisionTreeClassifier(max_depth=100)
model3.fit(X_train,Y_train)
ypredcate =model3.predict(X_test)

file_name3 ='Deciss_cweight.sav'
pickle.dump(model3,open(file_name3,'wb'))









######################################################################
# read test data
data = pd.read_csv('AppleStore_testing_classification.csv')

# show categories
categories = list(data["prime_genre"].unique())
# delete empty cells ,invalid data
data = remove_nulls(data)

data=data.drop(columns=['id','track_name','currency','ver'])
# show levels categouries
levels = list(data["rate"].unique())

data['cont_rating'] = data['cont_rating'].str.replace('+', '').astype(int)

data = pd.concat([ pd.get_dummies(data.prime_genre),data.drop('prime_genre', 1)], 1)

data['rating_count_tot'] = (data['rating_count_tot'] - min(data['rating_count_tot'])) / (
    max(data['rating_count_tot'] - min(data['rating_count_tot'])))
data['size_bytes'] = (data['size_bytes'] - min(data['size_bytes'])) / (
    max(data['size_bytes'] - min(data['size_bytes'])))
X2=pd.DataFrame(data,columns=X.columns)
X=X2.fillna(0)


data['rate']= data['rate'].str.replace('High', '2')
data['rate']= data['rate'].str.replace('Low', '0')
data['rate']= data['rate'].str.replace('Intermediate', '1')
data['rate']=data['rate'].astype(int)
X=data.iloc[:,:-1]
Y=data['rate']
fs = SelectKBest(score_func=f_classif, k=4)
X_selected = fs.fit_transform(X, Y)





Loaded_model=pickle.load(open('Deciss_cweight.sav','rb'))

result1=Loaded_model.score(X_selected,Y)
print (result1)



