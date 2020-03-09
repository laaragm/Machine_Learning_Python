#!/usr/bin/env python
# coding: utf-8

# In[7]:


#carregar a base de dados
import pandas as pd 
#divisão da base de dados
from sklearn.model_selection import train_test_split
#Transformar atributos categóricos em numéricos 
from sklearn.preprocessing import LabelEncoder
#matriz de confusão, calcular taxa de acertos
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
#visualização 
import graphviz
from sklearn.tree import export_graphviz
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

arvore = DecisionTreeClassifier() #obj
#Gerando a árvore
arvore.fit(x_treinamento, y_treinamento)
#visualizando a árvore - abra o .dot, copie o código
#cole no http://www.webgraphviz.com/ e clique em Generate Graph
#poderemos ver a árvore de decisão completa lá, que é efetivamente
#a aprendizagem desse algoritmo
export_graphviz(arvore, out_file='tree.dot')

#Fazendo as previsões: testando os  dados 30% separados para teste
#utilizando o modelo previamente gerado pela árvore de decisão
previsoes = arvore.predict(x_teste)
#Gerando a matriz de confusão -  frequências de classificação para cada classe do modelo
confusao = confusion_matrix(y_teste, previsoes)

#visualização da matriz de confusão bonitinha
visualizador = ConfusionMatrix(DecisionTreeClassifier())
visualizador.fit(x_treinamento, y_treinamento)
visualizador.score(x_teste, y_teste)
visualizador.poof()

indice_acertos = accuracy_score(y_teste, previsoes)
print(indice_acertos)
indice_erros = 1-indice_acertos


# In[ ]:




