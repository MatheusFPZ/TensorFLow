import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Carregar o arquivo CSV
nome_arquivo = 'punch_29.csv'  # Substitua pelo nome do seu arquivo CSV
dados = pd.read_csv(nome_arquivo)

# Extrair os 60 primeiros valores de cada coluna para X, Y e Z
valores_x = dados['x_acceleration'].values[:60]
valores_y = dados['y_acceleration'].values[:60]
valores_z = dados['z_acceleration'].values[:60]

# Organizar os dados em um array de entrada
dados_entrada = np.column_stack((valores_x, valores_y, valores_z))

# Determinar o rótulo do eixo com maior movimento (considerando o máximo valor de cada eixo)
rotulo_maior_movimento = np.argmax([np.max(valores_x), np.max(valores_y), np.max(valores_z)])

# Criar rótulos para cada amostra - para treinamento, precisamos de rótulos para cada amostra
# Neste caso, cada amostra terá o mesmo rótulo associado ao eixo com maior movimento
rotulos = np.full((60,), rotulo_maior_movimento)

# Inicializar o modelo (Random Forest Classifier)
modelo = RandomForestClassifier(random_state=42)

# Treinar o modelo
modelo.fit(dados_entrada, rotulos)

# Fazer previsões usando os mesmos dados do arquivo CSV (para este exemplo)
previsao = modelo.predict(dados_entrada)

# Determinar o eixo previsto com o maior movimento
eixo_maior_movimento = ['X', 'Y', 'Z'][int(previsao[0])]

print(f'O modelo prevê que o eixo com maior movimento é: {eixo_maior_movimento}')
