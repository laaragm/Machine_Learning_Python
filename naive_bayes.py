#!/usr/bin/env python
# coding: utf-8

# In[34]:


#carregar a base de dados
import pandas as pd 
#divisão da base de dados
from sklearn.model_selection import train_test_split
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
#conversão 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
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
#transformar os atributos categóricos em valores
#numéricos; GaussianNB não trabalha com esses atributos
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
#print(previsores[0]) - agora sim temos só dados numéricos
#e podemos passar essa base para o algoritmo naive bayes

#x_treinamento: 70% dados; x_teste:30% dados
#y_treinamento: resposta para os registros do treinamento
#y_teste: resposta para os registros do teste
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores,
    classe,
    test_size = 0.3, #30% teste; 70% treinamento
    random_state = 0) #sempre vai dividir a base de dados da mesma maneira

#Aprendizagem
naive_bayes = GaussianNB() #objeto naive_bayes
naive_bayes.fit(x_treinamento, y_treinamento)

#Previsão/Teste do modelo
previsoes = naive_bayes.predict(x_teste)
confusao = confusion_matrix(y_teste, previsoes)
#parâmetros: Ground truth (correct) labels; Predicted labels, as returned by a classifier
indice_acerto = accuracy_score(y_teste, previsoes)
indice_erro = 1-indice_acerto
print(indice_acerto)

visualizador = ConfusionMatrix(GaussianNB()) #criando objeto
visualizador.fit(x_treinamento, y_treinamento) 
visualizador.score(x_teste, y_teste) 
visualizador.poof() #renderiza a visualização

#--------------------------------------------------------------------------
#Simulando o modelo em Produção
novo_credito = pd.read_csv(r'/home/larag/Desktop/Data Science/Machine Learning/NovoCredit.csv')
novo_credito = novo_credito.iloc[:, 0:20].values
novo_credito[:,0] = labelencoder.fit_transform(novo_credito[:,0])
novo_credito[:,2] = labelencoder.fit_transform(novo_credito[:,2])
novo_credito[:,3] = labelencoder.fit_transform(novo_credito[:,3])
novo_credito[:,5] = labelencoder.fit_transform(novo_credito[:,5])
novo_credito[:,6] = labelencoder.fit_transform(novo_credito[:,6])
novo_credito[:,8] = labelencoder.fit_transform(novo_credito[:,8])
novo_credito[:,9] = labelencoder.fit_transform(novo_credito[:,9])
novo_credito[:,11] = labelencoder.fit_transform(novo_credito[:,11])
novo_credito[:,13] = labelencoder.fit_transform(novo_credito[:,13])
novo_credito[:,14] = labelencoder.fit_transform(novo_credito[:,14])
novo_credito[:,16] = labelencoder.fit_transform(novo_credito[:,16])
novo_credito[:,18] = labelencoder.fit_transform(novo_credito[:,18])
novo_credito[:,19] = labelencoder.fit_transform(novo_credito[:,19])

naive_bayes.predict(novo_credito)


# In[ ]:




