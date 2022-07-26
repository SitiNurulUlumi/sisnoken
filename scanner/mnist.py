import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from  tensorflow.keras.utils import to_categorical,plot_model 
import tensorflow as tf

mnist = tf.keras.datasets.mnist 
(xtrain, ytrain),(xtest, ytest) = mnist.load_data() 

xtrain=xtrain.reshape(60000,28,28,1)
xtest=xtest.reshape(10000,28,28,1)

xtrain=xtrain/255
xtest=xtest/255

ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)

from tensorflow.keras import models,layers
from keras.layers import Dense, Dropout, Flatten ,BatchNormalization

model=models.Sequential()
model.add(layers.Conv2D(filters=10,kernel_size=(2,2),input_shape=(28,28,1),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=12,kernel_size=(2,2),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=20,kernel_size=(2,2),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(150,activation='relu'))
model.add(layers.Dense(100,activation='relu'))
model.add(layers.Dense(50,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

history=model.fit(xtrain, ytrain, epochs=22,batch_size=1000,verbose=True,validation_data=(xtest, ytest))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.title("Showing the Train Vs Test Accuracy")
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(xtest,  ytest, verbose=2)

model.save('model_mnist.h5')
