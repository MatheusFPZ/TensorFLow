import tensorflow as tf
import numpy as np


modelo_lite_path = "modelo_lite.tflite"
interpreter = tf.lite.Interpreter(model_path=modelo_lite_path)
interpreter.allocate_tensors()
# Realizar inferência com dados de exemplo
input_data = np.array([[28.91	,-28.57	,5.72	,41.04


]], dtype=np.float32)
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
print("Resultado da inferência:", output_data)
if(output_data > 0.9):
    print("///////Soco///////")
else:
    print("////////Nao soco//////")