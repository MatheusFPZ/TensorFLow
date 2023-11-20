import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import time

# Carregar dados do CSV de treino
dados_treino = pd.read_csv('treino.csv')

# Remover a primeira linha do conjunto de treino (se necessário)
dados_treino = dados_treino.iloc[1:]

# Separar os recursos (X) e rótulos (y) do conjunto de treino
X_treino = dados_treino[['ax', 'ay', 'az', 'atotal']]
y_treino = dados_treino['soco']  # Substitua 'soco' pelo nome correto da coluna de rótulos

# Normalizar os dados de treino
scaler = StandardScaler()
X_treino = scaler.fit_transform(X_treino)

# Construir e treinar o modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(4,))
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
modelo.fit(X_treino, y_treino, epochs=100, batch_size=32, validation_split=0.2)

# Carregar dados do CSV de teste
dados_teste = pd.read_csv('teste4.csv')

# Remover a primeira linha do conjunto de teste (se necessário)
dados_teste = dados_teste.iloc[1:]

# Separar apenas os recursos (X) do conjunto de teste
X_teste = dados_teste[['ax', 'ay', 'az', 'atotal']]

# Normalizar os dados de teste
X_teste = scaler.transform(X_teste)

# Fazer previsões no conjunto de teste
previsoes = modelo.predict(X_teste)

cont =0
# Exibir as previsões
#print("Previsões:")
for i in range(len(previsoes)):
   if(previsoes[i]==1):
    cont= cont+1
    
    if(cont>=3):
        print("socou")
        cont=0
    else:
        print("//")
   else:
       print("//")

modelo.save('/home/linux/Área de Trabalho/modelos/modelo6')



#caminho_saida_csv = 'arquivo_saida2.csv'

# Criar um DataFrame com os rótulos previstos
#df_saida = pd.DataFrame({'Rotulos_Previstos': previsoes.flatten()})

# Salvar os rótulos previstos em um arquivo CSV
#df_saida.to_csv(caminho_saida_csv, index=False)