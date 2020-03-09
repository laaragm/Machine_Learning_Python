from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score

iris = datasets.load_iris() #carrega base
#print(iris.items())
unicos, quantidade = np.unique(iris.target, return_counts=True) #contagem elementos de cada classe
#print(unicos) #tem 3 elementos diferentes
#print(quantidade) #cada grupo tem 50 elementos

#Criando e treinando o modelo --------------------------------------------------------------------------
cluster = KMeans(n_clusters=3) #passar quantos clusters queremos definir - vamos dividir em 3 grupos
cluster.fit(iris.data) #formando os grupos

#define os centros de cada um dos grupos; teremos 3 centróides e cada um tem 3 atributos
#portanto, a variável 'centroides' nos retorna a média das instâncias(linhas) de cada atributo(coluna)
centroides = cluster.cluster_centers_
#print(centroides)
previsoes = cluster.labels_ #prevê o grupo onde cada instância vai se encaixar
#print(previsoes)

#Testando o modelo -------------------------------------------------------------------------------------
unicos2, quantidade2 = np.unique(previsoes, return_counts=True)
#print(unicos2) #quantidade de grupos - ainda 3 grupos (novos)
#print(quantidade2) #quantidade de elementos em cada grupo redefinida

resultados = confusion_matrix(iris.target, previsoes)
#print(resultados)


#visualizar o gráfico - passa o x e o y como parâmetros
#mostramos os elementos de cada grupo separadamente, por isso o filtro 'previsoes==' e etc
plt.scatter(iris.data[previsoes==0, 0], iris.data[previsoes==0, 1],
            c='blue', label='Sentosa')
plt.scatter(iris.data[previsoes==1, 0], iris.data[previsoes==1, 1],
            c='red', label='Versicolor')
plt.scatter(iris.data[previsoes==2, 0], iris.data[previsoes==2, 1],
            c='yellow', label='Virgica')
plt.legend()

indice_acertos = accuracy_score(iris.target, previsoes)
print(indice_acertos)