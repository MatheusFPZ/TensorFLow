import tensorflow as tf
import numpy as np

# Dados de entrada (exemplo)
# Suponha que são 10 linhas de dados com três características cada
dados = np.array([
    [5.18,	-13.68,	-3.17,	14.96],
[7.01,	-11.5,	-4.02,	14.05],
[9.87,	-10.03,	-4.04,	14.64],
[12.88,	-8.4,	-3.82,	15.84],
[14.24,	-6.46,	-2.63,	15.85],

    [-0.04,	-7.0,	1.64,	7.18],
[-0.36,	-6.08,	1.53,	6.27],
[-0.64,	-5.09,	1.32,	5.29],
[-0.72,	-4.57,	1.45,	4.84],
[-0.74,	-4.34,	1.42,	4.62]

])

# Saída desejada para cada linha (rótulos)
# Suponha que a coluna final representa se é um "soco" (True) ou não (False)
saidas = np.array([True, True, True, True, True, False, False, False, False, False])

# Convertendo os dados para tensores do TensorFlow
dados_tensor = tf.constant(dados, dtype=tf.float32)
saidas_tensor = tf.constant(saidas, dtype=tf.bool)

# Criando o modelo de classificação usando TensorFlow
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(4,))
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo
modelo.fit(dados_tensor, saidas_tensor, epochs=1000, verbose=0)

# Fazendo previsões com o modelo treinado
previsoes = modelo.predict(dados_tensor)

# Arredondando as previsões para valores binários (0 ou 1)
previsoes_binarias = np.round(previsoes).astype(bool)

cont=0
# Exibindo as previsões e rótulos reais
print("Saída Real     |     Previsão do Modelo")
for i in range(len(saidas)):
        print(f"{saidas[i]}              |     {previsoes_binarias[i][0]}")
        cont = previsoes_binarias[i][0]+cont
    
if(cont>=5):
            print("socou")
else:
            print("nada")

