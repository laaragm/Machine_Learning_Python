#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets #carregar bases de dados que já vem no sklearn
from scipy import stats #estatística
from yellowbrick.classifier import ConfusionMatrix

iris = datasets.load_iris()
#iris
stats.describe(iris.data)

previsores = iris.data
classe = iris.target

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                 classe,
                                                                 test_size=0.3,
                                                                 random_state=0)
#n_neighbors=3 considera os 3 vizinhos mais próximos
knn = KNeighborsClassifier(n_neighbors=3)
#Fazendo o treinamento 
knn.fit(x_treinamento, y_treinamento)
#Para fazer uma classificação simplesmente vai fazer a comparação da distância com
#esses registros que já estão armazenados
previsoes = knn.predict(x_teste)
confusao = confusion_matrix(y_teste, previsoes)
#visualização da matriz de confusão bonitinha
visualizador = ConfusionMatrix(KNeighborsClassifier(n_neighbors=3))
visualizador.fit(x_treinamento, y_treinamento)
visualizador.score(x_teste, y_teste)
visualizador.poof()

indice_acertos = accuracy_score(y_teste, previsoes)
print(indice_acertos)
indice_erros = 1-indice_acertos


# In[ ]:




