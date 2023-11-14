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
[-0.74,	-4.34,	1.42,	4.62],


])
dados_testes = np.array([
    [-1.62,	2.81,	-2.9,	4.35],
[-1.7,	3.16,	-3.0,	4.67],
[-1.75,	2.82,	-2.27,	4.02],
[-1.8,	2.35,	-2.0,	3.57],
[-1.78,	2.2,	-2.13,	3.54],
[-1.83,	2.13,	-1.67,	3.26],
[-2.11,	2.13,	-1.19,	3.22],
[-2.19,	2.2,	-0.95,	3.24],
[-2.25,	2.16,	-0.77,	3.21],
[-2.29,	2.11,	-0.56,	3.16],
[-2.44,	2.21,	-0.43,	3.32],
[-2.62,	2.43,	-0.64,	3.63]
])

# Saída desejada para cada linha (rótulos)
# Suponha que a coluna final representa se é um "soco" (True) ou não (False)
saidas = np.array([True, True, True, True, True, False, False, False, False, False])

# Convertendo os dados para tensores do TensorFlow
dados_tensor = tf.constant(dados, dtype=tf.float32)
saidas_tensor = tf.constant(saidas, dtype=tf.bool)
teste_sensor = tf.constant(dados_testes, dtype=tf.float32)

# Criando o modelo de classificação usando TensorFlow
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(4,))
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo
modelo.fit(dados_tensor, saidas_tensor, epochs=500, verbose=1)

# Fazendo previsões com o modelo treinado
#previsoes = modelo.predict(dados_tensor)
previsoes = modelo.predict(teste_sensor)

# Arredondando as previsões para valores binários (0 ou 1)
previsoes_binarias = np.round(previsoes).astype(bool)

cont=0
# Exibindo as previsões e rótulos reais
print("Saída Real     |     Previsão do Modelo")


print(previsoes)

modelo.save('/home/linux/Área de Trabalho/modelo5')