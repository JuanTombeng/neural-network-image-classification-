
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
import tensorflow_datasets as tfds


data = tfds.builder('rock_paper_scissors')
info = data.info
Train = tfds.load(name="rock_paper_scissors",split="train")
Test = tfds.load(name="rock_paper_scissors",split="test")
fig = tfds.show_examples(info, Train)
fig = tfds.show_examples(info, Test)



X = np.array([example['image'].numpy()[:,:,0] for example in Train])
Xlable = np.array([example['label'].numpy() for example in Train])

y = np.array([example['image'].numpy()[:,:,0] for example in Test])
ylable = np.array([example['label'].numpy() for example in Test])

X = X.reshape(2520,300,300,1)
y = y.reshape(372,300,300,1)

X = X.astype('float32')
y = y.astype('float32')

X /= 255
y /= 255

model = keras.Sequential([
  keras.layers.Flatten(),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dense(256, activation='relu'),
  keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(X, Xlable, epochs=5, batch_size=32)


while True:
    print("Enter a number")
    num = input()
    imagetest = y[int(num)].reshape(300,300)
    plt.imshow(imagetest, cmap='Greys_r')

    result = model.predict(np.array([y[int(num)]]))
    print(result)

    predicted_value = np.argmax(result)
    if predicted_value == 0:
        predicted_value == "Stone"
    elif predicted_value == 1:
        predicted_value == "Paper"
    else:
        predicted_value == "Scisorrs"
    print(predicted_value)

