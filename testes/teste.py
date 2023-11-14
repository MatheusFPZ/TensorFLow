import tensorflow as tf
import numpy as np



caminho_modelo_exportado = '/home/linux/Área de Trabalho/modelo5'

# Carregar o modelo
modelo_carregado = tf.keras.models.load_model(caminho_modelo_exportado)


dados_testes = np.array([
[-9.93,	0.75,	-0.73,	9.98],
[-7.51,	-7.13,	0.66,	10.37],
[-5.97,	-14.38,	02.05,	15.7],
[-3.3,	-18.95,	4.24,	19.69],
[02.08,	-23.29,	5.3,	23.97],
[10.79,	-27.26,	6.48,	30.02],
[20.89,	-29.62,	6.51,	36.82],
[28.91,	-28.57,	5.72,	41.04]
])

teste_sensor = tf.constant(dados_testes, dtype=tf.float32)


previsoes = modelo_carregado.predict(teste_sensor)
cont =0
for i in range(len(dados_testes)):
        print(f"{dados_testes[i]}              |     {previsoes[i][0]}")
        if(previsoes[i][0]>=0.5):
            cont = cont+1
        else:
            cont = cont+0

if(cont>=4):
            print("socou")
else:
            print("nada")