#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
#  @File           : RandomForestClassifier.py
#  @Author         : yiling.yang
#  @Date           : 2021/6/11
#  @Software       : PyCharm
#  @PythonVersion  :  
#  @Function       :  
# -------------------------------------------------------------------------------------------------
from config import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv(train_csv_path1)
train_set,test_set = train_test_split(train_data,test_size=0.1,random_state=10)
# print(test_set)

feature_train,target_train = train_data.drop('Survived',axis=1),train_data['Survived']
# feature_train,target_train = train_set.drop('Survived',axis=1),train_set['Survived']
feature_test,target_test = test_set.drop('Survived',axis=1),test_set['Survived']
# print(target_test)

#调参 n_estimators
# score_lt = []
# for i in range(1,301,10):
#     model = RandomForestClassifier(n_estimators=i
#                                 ,random_state=90)
#     model.fit(feature_train, target_train)
#     score = cross_val_score(model,feature_train,target_train,cv=5, scoring='accuracy').mean()
#     # score = model.score(feature_test , target_test)
#     score_lt.append(score)
#     print('精度：{},子树：{}'.format(score,i))
# score_max = max(score_lt)
# print('最高精度：{}'.format(score_max),
#       '子树数量为：{}'.format(score_lt.index(score_max)*10))
# # 绘制学习曲线
# x = np.arange(1,301,10)
# plt.subplot(111)
# plt.plot(x, score_lt, 'r-')
# plt.show()
#
score_lt = []
for i in range(5,21):
    model = RandomForestClassifier(n_estimators=i
                                ,random_state=90)
    model.fit(feature_train, target_train)
    score = cross_val_score(model,feature_train,target_train,cv=5, scoring='accuracy').mean()
    score_lt.append(score)
    print('精度：{},子树：{}'.format(score,i))
score_max = max(score_lt)
print('最高精度：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)+10))
# 绘制学习曲线
x = np.arange(5,21)
plt.subplot(111)
plt.plot(x, score_lt, 'r-')
plt.show()


model = RandomForestClassifier(n_estimators=16,random_state=90)
model.fit(feature_train,target_train)
# predict = model.predict(feature_test)
# score = model.score(feature_test , target_test) # 评估正确率
# print('Score = %f'%score)
feature = pd.read_csv(test_csv_path1)
test_tmp = pd.read_csv(test_csv_path)
predict = model.predict(feature)
# print(predict)
test_res= {'PassengerId':test_tmp['PassengerId'],'Survived':predict}
test_res = pd.DataFrame(test_res)
print(test_res)
test_res.to_csv(res_RFC_path,index=None)

