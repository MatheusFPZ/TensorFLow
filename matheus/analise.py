import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carregar dados do CSV
dados = pd.read_csv('analise.csv')

# Selecionar colunas relevantes
X = dados[['ax', 'ay', 'az', 'atotal']]

# Normalizar os dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar k-means
kmeans = KMeans(n_clusters=2, random_state=42)
dados['cluster'] = kmeans.fit_predict(X_scaled)
#dados.groupby('cluster').describe()

estatisticas_clusters = dados.groupby('cluster')[['ax', 'ay']].describe()

# Exibir as estatísticas descritivas
print(estatisticas_clusters)

# Visualizar os clusters
plt.scatter(dados['ax'], dados['ay'], c=dados['cluster'], cmap='viridis')
plt.title('Clusters')
plt.xlabel('ax')
plt.ylabel('ay')
plt.show()
