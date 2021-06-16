#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
#  @File           : data_cleaning_1.py
#  @Author         : yiling.yang
#  @Date           : 2021/6/9
#  @Software       : PyCharm
#  @PythonVersion  :  
#  @Function       :  
# -------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

from config import *

train_csv = pd.read_csv(train_csv_path)
test_csv = pd.read_csv(test_csv_path)
#合并csv文件，进行数据清洗
total_csv = [train_csv,test_csv]
total_csv = pd.concat(total_csv)
total_csv.to_csv(total_csv_path,index=None)

#初步分析及填充
print(train_csv.info()) #打印摘要
print(train_csv.describe()) #打印描述性统计信息
print(train_csv.isnull().sum())#打印train空值数量
print(test_csv.isnull().sum())#打印test空值数量 Fare有1个空值

age_mode = float(total_csv['Age'].dropna().mode())#年龄众数 #24
embarked_mode = str(total_csv['Embarked'].dropna().mode())# 启航港 S
fare_mean = float(total_csv['Fare'].dropna().mean()) #旅客票价平均数 #33.29

def replace_Age(x):
    if x == 'NA':
        x= age_mode
    if float(x)<=12: #婴少儿
        return 0;
    elif float(x)<=17: #青少年
        return 1
    elif float(x)<=45: #青壮年
        return 2
    else: #老年
        return 3
def replace_Sex(x):
    if x=="male":
        return 0
    else:
        return 1

def replace_Cabin(x):
    temp = ['Unknown','D', 'FG', 'B', 'C', 'A', 'F', 'EF', 'E', 'G', 'T']
    if x=='NA':
        return 0
    else:
        Letter = [chr(i) for i in range(65,91)]
        res = ''
        for i in Letter:
            if i in x:
                res += i
        return temp.index(res)

def replace_Embarked(x):
    if x == 'NA': #mode S
        return 1
    elif x=='Q':
        return 0
    elif x=='S':
        return 1
    else:
        return 2

def replace_Fare(x):
    if x == 'NA':
        x=fare_mean
    if float(x)<=10:
        return 0
    elif float(x)<=30:
        return 1
    elif float(x)<=50:
        return 2
    # elif float(x)<=100:
    #     return 3
    else:
        return 3
#特征工程
train_df = train_csv
test_df = test_csv
#Sex
train_df['Sex'] = train_df['Sex'].apply(replace_Sex)
test_df['Sex'] = test_df['Sex'].apply(replace_Sex)
#Age
train_df['Age'].fillna('NA',inplace=True)
train_df['Age'] = train_df['Age'].apply(replace_Age) #众数填充
test_df['Age'].fillna('NA',inplace=True)
test_df['Age'] = test_df['Age'].apply(replace_Age) #众数填充
#Cabin
train_df['Cabin'].fillna('NA',inplace=True)
train_df['Cabin'] = train_df['Cabin'].apply(replace_Cabin)
# data_list = set(list(train_df['Cabin'].apply(replace_Cabin)))
# print(data_list) #{'D', 'FG', 'B', 'C', 'Unknown', 'A', 'F', 'EF', 'E', 'G', 'T'}
test_df['Cabin'].fillna('NA',inplace=True)
test_df['Cabin'] = test_df['Cabin'].apply(replace_Cabin)
# data_list = set(list(test_df['Cabin'].apply(replace_Cabin)))
# print(data_list) #{'D', 'FG', 'B', 'Unknown', 'C', 'A', 'F', 'EF', 'E', 'G'}
#Embarked
train_df['Embarked'].fillna('NA',inplace=True)
train_df['Embarked'] = train_df['Embarked'].apply(replace_Embarked)
test_df['Embarked'] = test_df['Embarked'].apply(replace_Embarked)
# print(set(list(train_df['Embarked']))) #Q S C
#Fare
test_df['Fare'].fillna('NA',inplace=True)
test_df['Fare'] = test_df['Fare'].apply(replace_Fare)
train_df['Fare'] = train_df['Fare'].apply(replace_Fare)
# print(train_df['Fare'].describe())


#抛弃PassengerID Name Ticket
train_df = train_df.drop('PassengerId',axis=1)
train_df = train_df.drop('Name',axis=1)
test_df = test_df.drop('PassengerId',axis=1)
test_df = test_df.drop('Name',axis=1)
train_df = train_df.drop('Ticket',axis=1)
test_df = test_df.drop('Ticket',axis=1)


#保存
train_df.to_csv(train_csv_path1)
test_df.to_csv(test_csv_path1)








