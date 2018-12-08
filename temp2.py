# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import re
import numpy as np
#import matplotlib.pyplot as plt

#data_train = pd.read_csv("E:\\kaggle\\ttnk\\Train.csv")
#data_test = pd.read_csv("E:\\kaggle\\ttnk\\test.csv")
'''
data_train['Fare_bin'] = pd.qcut(data_train['Fare'], [0,10.5,75,513])
data_train['Fare_bin'].head()
'''
data_train_df = pd.read_csv("E:\\kaggle\\ttnk\\Train.csv")
data_test_df = pd.read_csv("E:\\kaggle\\ttnk\\test.csv")
data_test_df['Survived'] = 0
data_train_test = data_train_df.append(data_test_df)
PassengerId = data_test_df['PassengerId']
data_train_test['Embarked'].fillna(data_train_test['Embarked'].mode().iloc[0], inplace=True)

#姓名标准化
data_train_test['Title'] = data_train_test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
title_Dict = {}
title_Dict.update(dict.fromkeys(['Mr','Sir', 'Major', 'Dr', 'Rev','Jonkheer','capt'], 'Mr'))
title_Dict.update(dict.fromkeys(['Mme','mlle','Ms', 'Mrs', 'Lady', 'the Countess'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Master'], 'Master'))
data_train_test['Title'] = data_train_test['Title'].map(title_Dict)
#name的factorize化
data_train_test['Title'] = pd.factorize(data_train_test['Title'])[0]
#name的dummies化
#title_dummies_df = pd.get_dummies(data_train_test['Title'], prefix=data_train_test[['Title']].columns[0])
#data_train_test = pd.concat([data_train_test, title_dummies_df], axis=1)
#print(data_train_test.head())

#对Embarked进行填充并标准化，众数填充
data_train_test['Embarked'].fillna(data_train_test['Embarked'].mode().iloc[0], inplace=True)
data_train_test['Embarked'] = pd.factorize(data_train_test['Embarked'])[0]
#print(data_train_test.info())

#  Sex特征进行facrorizing
data_train_test['Sex'] = pd.factorize(data_train_test['Sex'])[0]
#print(data_train_test.info())

#对票价进行处理及标准化
data_train_test['Fare'] = data_train_test[['Fare']].fillna(data_train_test.groupby('Pclass').transform(np.mean))
fare_qujian=[0,1,10.5,75,513]
data_train_test['Fare'] = pd.cut(data_train_test['Fare'], fare_qujian)
data_train_test['Fare'] = pd.factorize(data_train_test['Fare'])[0]
#print(data_train_test.info())

#将Ticket分类标准化
data_train_test['Ticket_Number'] = data_train_test['Ticket'].str.split().str[0]
data_train_test['Ticket_Number'] = data_train_test['Ticket_Number'].apply(lambda x: 'U0' if x.isnumeric() else x)
data_train_test['Ticket_Number'] = pd.factorize(data_train_test['Ticket_Number'])[0]
#print(data_train_test)


#对已有年龄进行离散标准化
missing_age_df = pd.DataFrame(data_train_test[['Age', 'Embarked', 'Title','SibSp','Parch']])
missing_age_train_test = missing_age_df[missing_age_df['Age'].notnull()]

Age_qujian=[0,13,30,60,80]
missing_age_train_test['Age_bin'] = pd.cut(missing_age_train_test['Age'], Age_qujian)
#print(missing_age_train_test.head())
missing_age_train_test['Age_bin_id'] = pd.factorize(missing_age_train_test['Age_bin'])[0]
#Age_dummies_df = pd.get_dummies(missing_age_train_test['Age_bin'], prefix=missing_age_train_test['Age_bin'].columns[0])
#missing_age_df = pd.concat([missing_age_df, Age_dummies_df], axis=1)
missing_age_train = pd.DataFrame(missing_age_train_test[['Age_bin_id', 'Embarked', 'SibSp', 'Title', 'Parch']])
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
#print(missing_age_train.head())
#print(missing_age_test.head())

data_train_test.loc[(data_train_test.Age.notnull()), 'Age'] =missing_age_train_test['Age_bin_id']
print(data_train_test.head())


#对年龄进行预测
#from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age_bin_id'], axis=1)
    missing_age_Y_train = missing_age_train['Age_bin_id']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
    
    #随机森林
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [100], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=5, n_jobs=25, verbose=1)
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
    print('RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])

    #GDB
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [1000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=5, n_jobs=25, verbose=1)
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])
    
    #俩模型融合
    missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)
    
    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])

    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

    return missing_age_test

#填充age的预测值在data_train_test中
data_train_test.loc[(data_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)







#数据的备份
ttnk_data_train_test = data_train_test
data_train_test.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'],axis=1,inplace=True)


#数据的拆分
data_train = data_train_test[:891]
data_test = data_train_test[891:]



# LogisticRegression
def LogisticRegression_predict_Survived(data_train, data_test):
    from sklearn import linear_model
    ttnk_data_train_X = data_train.drop(['Survived'],axis=1)
    ttnk_data_train_Y = data_train['Survived']
    ttnk_data_test_X = data_test.drop(['Survived'],axis=1)
    #from sklearn.ensemble import RandomForestClassifier
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(ttnk_data_train_X, ttnk_data_train_Y)
    predictions = clf.predict(ttnk_data_test_X)
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("E:\\kaggle\\ttnk\\logistic_regression_predictions.csv", index=False)


# RandomForest    
def RandomForest_predict_Survived(data_train, data_test):
    from sklearn.ensemble import RandomForestRegressor
    ttnk_data_train_X = data_train.drop(['Survived'],axis=1)
    ttnk_data_train_Y = data_train['Survived']
    ttnk_data_test_X = data_test.drop(['Survived'],axis=1)
    rf_est = RandomForestRegressor(random_state=0 , min_samples_split= 3 , max_depth = 20, n_jobs=25)
    rf_est.fit(ttnk_data_train_X, ttnk_data_train_Y)
    #print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    #print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    #print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    predictions = rf_est.predict(ttnk_data_test_X)
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("E:\\kaggle\\ttnk\\RandomForest_predictions.csv", index=False)

    

#模型融合
from sklearn.ensemble import BaggingRegressor
def BaggingRegressor_predictions(data_train, data_test):
    ttnk_data_train_X = data_train.drop(['Survived'],axis=1)
    ttnk_data_train_Y = data_train['Survived']
    ttnk_data_test_X = data_test.drop(['Survived'],axis=1)
    clf = RandomForestRegressor(random_state=0)
    br_est =  BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
    br_est.fit(ttnk_data_train_X, ttnk_data_train_Y)
    predictions = br_est.predict(ttnk_data_test_X)
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("E:\\kaggle\\ttnk\\BaggingRegressor_predictions.csv", index=False)
