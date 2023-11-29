import pandas as pd
import numpy as np
from tensorflow.keras import layers, models

# Carregar os dados do arquivo CSV
data = pd.read_csv('punch_8modificado.csv')

# Separar os dados em eixos X, Y e Z
x_data = data[['x_acceleration']].to_numpy()  # Substitua 'x_column_name' pelo nome da coluna correspondente ao eixo X
y_data = data[['y_acceleration']].to_numpy()  # Substitua 'y_column_name' pelo nome da coluna correspondente ao eixo Y
z_data = data[['z_acceleration']].to_numpy()  # Substitua 'z_column_name' pelo nome da coluna correspondente ao eixo Z

# Combine os dados dos três eixos para formar as amostras
combined_data = np.stack((x_data, y_data, z_data), axis=-1)  # Combine os dados dos eixos X, Y e Z

# Reformatar os dados para ter uma dimensão 3D (amostras, tempo, eixos)
combined_data = combined_data.reshape(-1, 60, 3)

# Rótulos para os dados
acao_labels = data['soco'].to_numpy()  # Substitua 'acao' pelo nome da coluna correspondente aos rótulos

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


data_test = pd.read_csv('punch_8.csv')

# Separar os dados de teste em eixos X, Y e Z
x_data_test = data_test[['x_acceleration']].to_numpy()  # Substitua 'x_acceleration' pelo nome da coluna correspondente ao eixo X
y_data_test = data_test[['y_acceleration']].to_numpy()  # Substitua 'y_acceleration' pelo nome da coluna correspondente ao eixo Y
z_data_test = data_test[['z_acceleration']].to_numpy()  # Substitua 'z_acceleration' pelo nome da coluna correspondente ao eixo Z


# Combine os dados dos três eixos para formar as amostras de teste
combined_data_test = np.stack((x_data_test, y_data_test, z_data_test), axis=-1)  # Combine os dados dos eixos X, Y e Z
combined_data_test = combined_data_test.reshape(-1, 60, 3)  # Reformatar os dados para ter uma dimensão 3D (amostras, tempo, eixos)

# Avaliar o modelo nos dados de teste
previsao = modelo.predict(combined_data_test)

print(previsao)
classe_prevista = np.argmax(previsao)

# Mostrar a classe prevista
print("Classe prevista:", classe_prevista)