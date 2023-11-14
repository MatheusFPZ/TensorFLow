
import tensorflow as tf


# Carregue o modelo SavedModel
saved_model_dir = "/home/linux/Área de Trabalho/modelos/modelo6"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Opções de otimização (opcional)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Converter o modelo para TensorFlow Lite
modelo_lite = converter.convert()

# Salve o modelo TensorFlow Lite em um arquivo .tflite
with open("modelo_lite2.tflite", "wb") as f:
    f.write(modelo_lite)
