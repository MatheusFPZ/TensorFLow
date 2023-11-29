import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model



dados_teste = pd.read_csv('punch_29.csv')

# Processar os dados de teste da mesma maneira que fez durante o treinamento
# Certifique-se de realizar as mesmas etapas de pré-processamento, como normalização, redimensionamento, etc.

# Separar os dados de teste em eixos X, Y, Z (dependendo de como você os processou durante o treinamento)
x_teste = dados_teste['x_acceleration'].to_numpy()  
y_teste = dados_teste['y_acceleration'].to_numpy()  
z_teste = dados_teste['z_acceleration'].to_numpy()  

# Agrupar os dados em sequências do mesmo tamanho usado durante o treinamento (por exemplo, sequências de 3 por 30)
tamanho_sequencia = 60
grupos_sequenciais_teste = []

for i in range(0, len(x_teste), tamanho_sequencia):
    if i + tamanho_sequencia <= len(x_teste):
        sequencia_x = x_teste[i:i + tamanho_sequencia]
        sequencia_y = y_teste[i:i + tamanho_sequencia]
        sequencia_z = z_teste[i:i + tamanho_sequencia]
        grupos_sequenciais_teste.append([sequencia_x, sequencia_y, sequencia_z])

# Converter para numpy array
grupos_sequenciais_teste = np.array(grupos_sequenciais_teste)

# Carregar o modelo treinado
modelo = load_model('/home/linux/Área de Trabalho/modelos/modelo9')

# Fazer previsões nos dados de teste
previsoes = modelo.predict(grupos_sequenciais_teste)

# Exibir as previsões
print(previsoes)