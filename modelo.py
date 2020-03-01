# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 22:22:46 2020

@author: diego
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import seaborn as sns

# PARA ESTUDIANTES DE PORTUGUÉS

students_por = pd.read_csv(
        "C:/Users/diego/OneDrive/Documentos/UNAM/Cursos/Data Mining/student/student-por.csv", 
        sep=";")

# obtener columnas categoricas
s = (students_por.dtypes == 'object')
object_cols = list(s[s].index)

# aplicamos el encoder
label_encoder = LabelEncoder()
for col in object_cols:
    students_por[col] = label_encoder.fit_transform(students_por[col])

# Se guarda el dataset
#students_por.to_csv('students_por_transform.csv')


# PARA ESTUDIANTES DE MATE

students_mat = pd.read_csv(
        "C:/Users/diego/OneDrive/Documentos/UNAM/Cursos/Data Mining/student/student-por.csv", 
        sep=";")

# obtener columnas categoricas
sm = (students_mat.dtypes == 'object')
object_cols_mat = list(sm[sm].index)

# aplicamos el encoder
label_encoder = LabelEncoder()
for col in object_cols_mat:
    students_mat[col] = label_encoder.fit_transform(students_mat[col])

# Se guarda el dataset
#students_por.to_csv('students_mat_transform.csv')


students_total = pd.concat([students_mat, students_por])

#sns.swarmplot(x=students_total['school'],
#              y=students_total['G3'], hue=students_total['sex'])


# comprobar si hay nulos
cols_with_missing = [col for col in students_total.columns
                     if students_total[col].isnull().any()]

features_to_remove = ['G1', 'G2', 'Medu', 'Walc', 'freetime', 'activities', 
                      'Pstatus', 'guardian', 'age', 'higher', 'reason', 'address']

Y = students_total.G3

features_to_remove.append('G3')

X = students_total.drop(features_to_remove, axis=1)
X.head()

train_X, val_X, train_y, val_y = train_test_split(X, Y, test_size=0.25)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
students_pred = forest_model.predict(val_X)
print("Sin cross validation el score es ", mean_absolute_error(val_y, students_pred))

print("Con cross validation la precisión es ", 
      (cross_val_score(forest_model, X, Y, cv=5).mean())*100)


comparation = pd.DataFrame({
        "val_y": val_y.to_list(),
        "prediction": students_pred},
        columns=['val_y', "prediction"])


sns.regplot(x=comparation.prediction, y=comparation.val_y)






