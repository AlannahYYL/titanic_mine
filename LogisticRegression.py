#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
#  @File           : LogisticRegression.py
#  @Project        : titanic_mine
#  @Author         : yiling.yang
#  @Date           : 2021/6/16
#  @Software       : PyCharm
#  @PythonVersion  :  
#  @Function       :  
# -------------------------------------------------------------------------------------------------
from config import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,accuracy_score

train_data = pd.read_csv(train_csv_path1)
train_set,test_set = train_test_split(train_data,test_size=0.1,random_state=10)
# print(test_set)

feature_train,target_train = train_data.drop('Survived',axis=1),train_data['Survived']
# feature_train,target_train = train_set.drop('Survived',axis=1),train_set['Survived']
feature_test,target_test = test_set.drop('Survived',axis=1),test_set['Survived']
# print(target_test)

model = LogisticRegression(max_iter=1000)
model.fit(feature_train,target_train)
# #查看模型精度
# predictions = model.predict(feature_test)
# score = cross_val_score(model,feature_train,target_train,cv=5, scoring='accuracy').mean()
# print('score:',score)
#预测
feature = pd.read_csv(test_csv_path1)
test_tmp = pd.read_csv(test_csv_path)
predict = model.predict(feature)
test_res= {'PassengerId':test_tmp['PassengerId'],'Survived':predict}
test_res = pd.DataFrame(test_res)
print(test_res)
test_res.to_csv(res_LR_path,index=None)
