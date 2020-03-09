#!/usr/bin/env python
# coding: utf-8

# In[11]:


#carregar a base de dados
import pandas as pd 
#divisão da base de dados
from sklearn.model_selection import train_test_split
#Transformar atributos categóricos em numéricos 
from sklearn.preprocessing import LabelEncoder
#matriz de confusão, calcular taxa de acertos
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
#visualização de modelos de machine learning
from yellowbrick.classifier import ConfusionMatrix

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

#vai gerar 100 árvores de decisão
floresta = RandomForestClassifier(n_estimators=100)
floresta.fit(x_treinamento, y_treinamento)
previsoes = floresta.predict(x_teste)
confusao = confusion_matrix(y_teste, previsoes)

visualizador = ConfusionMatrix(RandomForestClassifier())
visualizador.fit(x_treinamento, y_treinamento)
visualizador.score(x_teste, y_teste)
visualizador.poof()

indice_acertos = accuracy_score(y_teste, previsoes)
print(indice_acertos)
indice_erros = 1-indice_acertos
#floresta.estimators_ #mostra as árvores que foram criadas
#floresta.estimators_[2] #mostra a terceira árvore gerada


# In[ ]:




