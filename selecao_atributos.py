#!/usr/bin/env python
# coding: utf-8

# In[21]:


#carregar a base de dados
import pandas as pd 
import sklearn
from sklearn import svm, preprocessing
#divisão da base de dados
from sklearn.model_selection import train_test_split
#Transformar atributos categóricos em numéricos 
from sklearn.preprocessing import LabelEncoder
#matriz de confusão, calcular taxa de acertos
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
#visualização de modelos de machine learning
from yellowbrick.classifier import ConfusionMatrix
from sklearn.ensemble import ExtraTreesClassifier

#abre o arquivo .csv especificando o caminho
credito = pd.read_csv(r'/home/larag/Desktop/Data Science/Machine Learning/Credit.csv')

#pegando os valores de todas as linhas(instâncias) da base de dados
#e todas as colunas (menos class)
#.iloc[] is primarily integer position based (from 0 to length-1 of the axis), 
#but may also be used with a boolean array
previsores = credito.iloc[:, 0:20].values
classe = credito.iloc[:, 20].values

#tratando os dados
#transformar os atributos categóricos em numéricos
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder.fit_transform(previsores[:,6])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder.fit_transform(previsores[:,9])
previsores[:,11] = labelencoder.fit_transform(previsores[:,11])
previsores[:,13] = labelencoder.fit_transform(previsores[:,13])
previsores[:,14] = labelencoder.fit_transform(previsores[:,14])
previsores[:,16] = labelencoder.fit_transform(previsores[:,16])
previsores[:,18] = labelencoder.fit_transform(previsores[:,18])
previsores[:,19] = labelencoder.fit_transform(previsores[:,19])

#divisão dos dados em treinamento(70%) e teste(30%)
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                 classe,
                                                                 test_size=0.3,
                                                                 random_state=0)

#parâmetro colocado só pra não dar future warning
modelo = SVC(gamma='scale')
#Treina e gera o modelo
modelo.fit(x_treinamento, y_treinamento)
#Aplicando no modelo os dados reservados para teste(30%)
previsoes = modelo.predict(x_teste)
#Calcula a taxa de acertos
indice_acertos = accuracy_score(y_teste, previsoes)
print(indice_acertos)

#parâmetro colocado só pra não dar future warning
forest = ExtraTreesClassifier(n_estimators=100)
forest.fit(x_treinamento, y_treinamento)
#calcula os atributos mais importantes
importancias = forest.feature_importances_
#print(importancias) 

#------------------------------------------------------------------------
#Selecionando apenas 4 dos atributos mais importantes para gerar a árvore
x_treinamento_novo = x_treinamento[:, [0,1,2,3]]
x_teste_novo = x_teste[:, [0,1,2,3]]
modelo_novo = SVC(gamma='scale')
#Passando novos dados a serem treinados(saída - y_treinamento continua a mesma)
modelo_novo.fit(x_treinamento_novo, y_treinamento)
#Testando o modelo novo
previsoes_novas = modelo_novo.predict(x_teste_novo)
indice_acertos_novo = accuracy_score(y_teste, previsoes_novas)
print('Novo índice de acertos após ajuste do modelo: ', indice_acertos_novo)


# In[ ]:




