import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
data = pd.read_csv('punch_8.csv')

# Exibir as primeiras linhas do arquivo CSV para verificar o formato dos dados
print(data.head())

# Separar os dados em X, Y e Z
x_data = data[['x_acceleration']].to_numpy()  # Substitua 'x_column_name' pelo nome da coluna correspondente ao eixo X
y_data = data[['y_acceleration']].to_numpy()  # Substitua 'y_column_name' pelo nome da coluna correspondente ao eixo Y
z_data = data[['z_acceleration']].to_numpy()  # Substitua 'z_column_name' pelo nome da coluna correspondente ao eixo Z

# Combine os dados dos três eixos para formar as amostras
combined_data = np.stack((x_data, y_data, z_data), axis=-1)  # Combine os dados dos eixos X, Y e Z

# Rótulos para os dados
# Dependendo do seu arquivo CSV, você precisa definir os rótulos de alguma maneira
#labels = data['label_column'].to_numpy()  # Substitua 'label_column' pelo nome da coluna dos rótulos

# Reformatar os dados para ter uma dimensão 3D (amostras, tempo, eixos)
combined_data = combined_data.reshape(-1, 30, 3)

# Visualize um exemplo de gráfico para cada eixo (X, Y, Z)
exemplo_grafico = combined_data[0]  # Exemplo do primeiro gráfico

fig, axs = plt.subplots(3)
fig.suptitle('Exemplo de gráfico para um dos eixos (X, Y, Z)')

for i, eixo in enumerate(exemplo_grafico.T):
    axs[i].plot(eixo)
    axs[i].set_ylabel(f'Eixo {i+1}')

plt.xlabel('Tempo')
plt.show()

# Construir o modelo CNNa
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(30, 3)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(combined_data, epochs=10, batch_size=32, validation_split=0.2, shuffle=True)
