import tensorflow as tf
import csv  
print("TensorFlow version:", tf.__version__)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_df = pd.read_csv('train.csv.zip',compression='zip')
x_train, y_train = train_df.values[:, 1: 785] / 255.0, train_df.values[:, 0]
test_df = pd.read_csv('test.csv.zip',compression='zip')
x_test = test_df.values / 255.0

model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(input_shape=(784)),
  tf.keras.layers.Dense(input_shape = (784, ), units = 256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=32)


probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

header = ["ImageId", "Label"]
with open('predictions.csv', 'w') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)
    for ImageId, prediction in enumerate(x_test):
      data = [ImageId + 1, np.argmax(probability_model(np.reshape(x_test[ImageId], (1, 784))))]
      writer.writerow(data)

