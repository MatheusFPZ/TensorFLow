import pandas as pd
import numpy as np
from tensorflow.keras import layers, models

# Carregar os dados do arquivo CSV
data = pd.read_csv('punch_8modificado.csv')

# Exibir as primeiras linhas do arquivo CSV para verificar o formato dos dados
print(data.head())

# Separar os dados em eixos X, Y, Z e a coluna de rótulos (soco)
x_data = data['x_acceleration'].to_numpy()  # Substitua 'x_acceleration' pelo nome da coluna correspondente ao eixo X
y_data = data['y_acceleration'].to_numpy()  # Substitua 'y_acceleration' pelo nome da coluna correspondente ao eixo Y
z_data = data['z_acceleration'].to_numpy()  # Substitua 'z_acceleration' pelo nome da coluna correspondente ao eixo Z
labels = data['soco'].to_numpy()  # Substitua 'soco' pelo nome da coluna correspondente aos rótulos

# Definir rótulos manualmente com base em algum critério
# Por exemplo, atribuir rótulos 1 para sequências que contenham um soco e 0 para sequências que não contenham
# Aqui, estou considerando um valor de 1 na coluna 'soco' como indicativo de que houve um soco
# Você pode ajustar isso de acordo com a lógica ou critério do seu problema
rótulos_definidos = np.where(labels == 1, 1, 0)

# Separar os dados em grupos de 3 por 30 para cada eixo
tamanho_sequencia = 60
grupos_sequenciais = []

for i in range(0, len(x_data), tamanho_sequencia):
    if i + tamanho_sequencia <= len(x_data):
        sequencia_x = x_data[i:i + tamanho_sequencia]
        sequencia_y = y_data[i:i + tamanho_sequencia]
        sequencia_z = z_data[i:i + tamanho_sequencia]
        grupos_sequenciais.append([sequencia_x, sequencia_y, sequencia_z])

# Converter para numpy array
grupos_sequenciais = np.array(grupos_sequenciais)

# Verificar a forma dos dados
print(grupos_sequenciais.shape)  # Saída deve ser (n_sequencias, 3, tamanho_sequencia)
print(rótulos_definidos)  # Rótulos definidos para as sequências

model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(3, 60)),
    layers.MaxPooling1D(pool_size=1),  # Ajuste aqui
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Camada de saída para classificação binária
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(grupos_sequenciais, rótulos_definidos, epochs=30, batch_size=32, validation_split=0.2, shuffle=True)


model.save('/home/linux/Área de Trabalho/modelos/modelo9')
