import tensorflow as tf

model = tf.keras.models.load_model("model/model_uang.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open("model/model_uang.tflite", "wb") as f:
    f.write(tflite_model)

print("Berhasil membuat model_uang.tflite")