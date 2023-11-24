import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Criar os dados para "soco" e "não soco"
soco_data = np.array([
    [-0.6003418,1.178711,-0.4172363],
    [-0.5666504,1.426514,-0.7111816],
    [-0.6679688,1.743408,-0.9182129]
    # ... (adicionar todas as linhas de dados de "soco" aqui)
])

nao_soco_data = np.array([
    [-0.4760742, -0.6408691, -0.1665039],
    [-0.4760742, -0.6408691, -0.1665039],
    [-0.4760742, -0.6408691, -0.1665039],
    [-0.4760742, -0.6408691, -0.1665039],
    # ... (adicionar todas as linhas de dados de "não soco" aqui)
])

# Criar rótulos para os dados
soco_labels = np.ones(len(soco_data))  # Definir rótulos 1 para "soco"
nao_soco_labels = np.zeros(len(nao_soco_data))  # Definir rótulos 0 para "não soco"

# Juntar todos os dados e rótulos
all_data = np.vstack((soco_data, nao_soco_data))
all_labels = np.concatenate((soco_labels, nao_soco_labels))

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construir o modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária para classificação
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Avaliar o modelo
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')
