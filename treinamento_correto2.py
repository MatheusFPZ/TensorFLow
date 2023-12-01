import pandas as pd
import numpy as np
from tensorflow.keras import layers, models

# Carregar os dados do arquivo CSV
data = pd.read_csv('punch_8modificado.csv')

# Separar os dados em eixos X, Y e Z
x_data = data[['x_acceleration']].to_numpy()
y_data = data[['y_acceleration']].to_numpy()
z_data = data[['z_acceleration']].to_numpy()

# Calcular a variação em cada eixo ao longo do tempo (60 timestamps)
var_x = np.var(x_data)
var_y = np.var(y_data)
var_z = np.var(z_data)

# Determinar o eixo com a maior variação
maior_variacao = max(var_x, var_y, var_z)
if maior_variacao == var_x:
    eixo_maior_variacao = 'X'
elif maior_variacao == var_y:
    eixo_maior_variacao = 'Y'
else:
    eixo_maior_variacao = 'Z'

print(f"Eixo com maior variação: {eixo_maior_variacao} ({maior_variacao})")

# Combine os dados dos três eixos para formar as amostras
combined_data = np.stack((x_data, y_data, z_data), axis=-1)
combined_data = combined_data.reshape(-1, 60, 3)

# Rótulos para os dados
acao_labels = data['soco'].to_numpy()

# Criar um modelo com uma única saída para prever 'acao'
modelo = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(60, 3)),
    layers.MaxPooling1D(pool_size=1),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # Camada de saída com 3 unidades para prever 'acao'
])

# Compilar o modelo
# Utilize 'sparse_categorical_crossentropy' como função de perda para lidar com rótulos inteiros
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo com os rótulos mapeados para 'acao'
modelo.fit(combined_data, acao_labels, epochs=60, batch_size=32, validation_split=0.2, shuffle=True)
