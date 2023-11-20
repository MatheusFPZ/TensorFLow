import tensorflow as tf
import numpy as np

# Carregar o modelo TensorFlow Lite
modelo_lite_path = "modelo_lite.tflite"
interpreter = tf.lite.Interpreter(model_path=modelo_lite_path)
interpreter.allocate_tensors()

# Simulação de dados do acelerômetro (substitua isso pelos seus próprios dados)
dados_do_acelerometro = [
    [-0.8662109, 1.318359, -0.2263184],
    [-1.021484, -0.09106445, 0.5639648],
    [-0.9589844, -0.140625, 0.7404785],
   [ -0.8095703,	1.267578,	-0.4123535],
   [ -0.8662109	,1.318359	,-0.2263184],
  [  -0.8620605	,1.174316,	-0.359375]

    # Adicione mais linhas de dados conforme necessário
]
cont=0
# Para cada linha de dados do acelerômetro
for linha in dados_do_acelerometro:
    input_data = np.array([linha], dtype=np.float32)  # Formatar a linha como um array numpy

    # Definir o tensor de entrada do modelo TFLite
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Realizar a inferência
    interpreter.invoke()

    # Obter a saída
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Processar a saída (substitua isso pelo processamento adequado para o seu caso)

    
    if output_data >= 0.5:
        cont =cont+1
        
    else:
        cont= cont+0

if cont>=3:
    print("Socou")
else:
    print("Nao socou")