import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar dados do CSV
dados = pd.read_csv('treino.csv')
dados_teste = pd.read_csv('teste2.csv')

# Separar os recursos (X) e rótulos (y)
X = dados[['ax', 'ay', 'az', 'atotal']]
y = dados['soco']  # Substitua 'rotulo' pelo nome real da coluna que indica soco ou não soco

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Compilar o modelo com a entropia cruzada binária como função de perda
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Avaliar o modelo
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

#model.save('/home/linux/Área de Trabalho/modelo4')
model.predict()