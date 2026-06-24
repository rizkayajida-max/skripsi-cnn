import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sns


# Lokasi dataset
train_dir = "dataset_fix/train"
test_dir = "dataset_fix/test"

# Parameter
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20

# Load dataset train
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Load dataset test
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Nama kelas
class_names = train_dataset.class_names
print("Kelas:", class_names)

# Normalisasi
normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# Model CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(4, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Ringkasan model
model.summary()

# Training
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset
)

# Buat folder hasil jika belum ada
os.makedirs("hasil", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Simpan model
model.save("model/model_uang.keras")

# Evaluasi
loss, accuracy = model.evaluate(test_dataset)

print(f"\nAkurasi Testing: {accuracy*100:.2f}%")

# =====================================
# Classification Report Training
# =====================================

y_true_train = []
y_pred_train = []

for images, labels in train_dataset:

    predictions = model.predict(images, verbose=0)

    y_true_train.extend(
        np.argmax(labels.numpy(), axis=1)
    )

    y_pred_train.extend(
        np.argmax(predictions, axis=1)
    )

print("\n=== Classification Report Training ===")

print(
    classification_report(
        y_true_train,
        y_pred_train,
        target_names=class_names
    )
)

# Menyimpan label asli dan prediksi

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    y_true.extend(
        np.argmax(labels.numpy(), axis=1)
    )

    y_pred.extend(
        np.argmax(predictions, axis=1)
    )

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix CNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("hasil/confusion_matrix.png")

plt.show()

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix CNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("hasil/confusion_matrix.png")

plt.show()

# Classification Report

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

# Grafik Accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Grafik Accuracy CNN')
plt.savefig('hasil/grafik_accuracy.png')
plt.show()

# Grafik Loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Grafik Loss CNN')
plt.savefig('hasil/grafik_loss.png')
plt.show()