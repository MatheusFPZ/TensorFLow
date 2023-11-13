import pandas as pd
import numpy as np
import tensorflow as tf

# Caminho onde o modelo foi exportado
caminho_modelo_exportado = '/home/linux/Área de Trabalho/modelo4'

# Carregar o modelo
modelo_carregado = tf.keras.models.load_model(caminho_modelo_exportado)

# Carregar os dados de teste do CSV (substitua 'seu_arquivo_teste.csv' pelo nome real do seu arquivo)
caminho_arquivo_teste = 'teste2.csv'
dados_teste = pd.read_csv(caminho_arquivo_teste)

# Certifique-se de que os dados de teste estejam no mesmo formato que o modelo espera
# Isso pode incluir a seleção de recursos, normalização, etc.

# Supondo que 'novos_dados' seja o conjunto de dados que você deseja testar
# Certifique-se de que 'novos_dados' esteja no mesmo formato que o modelo espera
# Aqui, estou usando todas as colunas exceto a primeira como características
novos_dados = dados_teste.iloc[:, 1:]

# Realizar previsões
novas_previsoes = modelo_carregado.predict(novos_dados)

# Se for um problema de classificação binária com saída sigmoid, você pode arredondar para 0 ou 1
rotulos_previstos = (novas_previsoes > 0.5).astype(int)

# Imprimir ou usar os rótulos previstos conforme necessário
#print(rotulos_previstos)
caminho_saida_csv = 'arquivo_saida.csv'

# Criar um DataFrame com os rótulos previstos
df_saida = pd.DataFrame({'Rotulos_Previstos': rotulos_previstos.flatten()})

# Salvar os rótulos previstos em um arquivo CSV
df_saida.to_csv(caminho_saida_csv, index=False)
