from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, SimpleRNN
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import RMSprop

import numpy as np 
import matplotlib.pyplot as plt 
import warnings

from tensorflow.python.ops.gen_math_ops import mod
warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#레이블 개수 세기
num_labels = len(np.unique(y_train)) #0~9 10개 

unique, counts = np.unique(y_train, return_counts=True)
print("Train lables : ", dict(zip(unique, counts)))
unique, counts = np.unique(y_test, return_counts=True)
print("Test lables : ", dict(zip(unique, counts)))

#샘플 추출
indexes = np.random.randint(0, x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5, 5, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.show()
plt.savefig("mnist-samples.png")
plt.close("all")


#one-hot vector encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#이미지 차원 -> 정사각형으로 가정
image_size = x_train.shape[1]
input_size = image_size * image_size

#크기 조정, 정규화
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

#신경망 매개변수
batch_size = 128
hidden_units = 256
dropout = 0.45

#MLP (각 layer 다음에는 ReLU, dropout 적용)
model_MLP = Sequential()
model_MLP.add(Dense(hidden_units, input_dim=input_size))
#비선형 절차 -> relu 활성화 삽입으로써 비선형 매핑
model_MLP.add(Activation('relu'))  #Relu(정류 선형 유닛)
model_MLP.add(Dropout(dropout))
model_MLP.add(Dense(hidden_units))
model_MLP.add(Activation('relu'))
model_MLP.add(Dropout(dropout))
model_MLP.add(Dense(num_labels))

#one-hot vector print
model_MLP.add(Activation('softmax'))
print(model_MLP.summary())

#one-hot vector 손실함수 - adam 최적화 사용, 분류 작업 지표 : 정확도
model_MLP.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#훈련
model_MLP.fit(x_train, y_train, epochs=20, batch_size=batch_size)

#검증
loss, acc = model_MLP.evaluate(x_test, y_test, batch_size=batch_size)
print("\n Test Accuracy %.1f%%" %(100.0 * acc))

#CNN
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]

x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


#신경망 매개변수
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.2

#모델은 CNN-ReLU-MaxPooling
model_CNNN1 = Sequential()
model_CNNN1.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
model_CNNN1.add(MaxPooling2D(pool_size))
model_CNNN1.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
model_CNNN1.add(MaxPooling2D(pool_size))
model_CNNN1.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
model_CNNN1.add(Flatten())

#정규화로 드롭아웃 추가
model_CNNN1.add(Dropout(dropout))

#출력 계층은 10개 요소로 구성된 one-hot vector
model_CNNN1.add(Dense(num_labels))
model_CNNN1.add(Activation('softmax'))
print(model_CNNN1.summary())

model_CNNN1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_CNNN1.fit(x_train, y_train, epochs=10, batch_size=batch_size)

_, acc = model_CNNN1.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("\n Test accuracy: %.1f%%" % (100.0 * acc))

model_CNNN2 = Sequential()
model_CNNN2.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
model_CNNN2.add(MaxPooling2D(pool_size))
model_CNNN2.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
model_CNNN2.add(MaxPooling2D(pool_size))
model_CNNN2.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
model_CNNN2.add(Flatten())

model_CNNN2.add(Dropout(dropout))

model_CNNN2.add(Dense(num_labels))
model_CNNN2.add(Activation('softmax'))
print(model_CNNN2.summary())

model_CNNN2.compile(loss='categorical_crossentropy', optimizer=RMSprop(),metrics=['accuracy'])

model_CNNN2.fit(x_train, y_train, epochs=10, batch_size=batch_size)

_, acc = model_CNNN2.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("\n Test accuracy: %.1f%%" % (100.0 * acc))

#RNN 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size])
x_test = np.reshape(x_test,[-1, image_size, image_size])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (image_size, image_size)
batch_size = 128
units = 256
dropout = 0.2

model_RNN = Sequential()
model_RNN.add(SimpleRNN(units=units, dropout=dropout, input_shape=input_shape))
model_RNN.add(Dense(num_labels))
model_RNN.add(Activation('softmax'))
print(model_RNN.summary())

model_RNN.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model_RNN.fit(x_train, y_train, epochs=20, batch_size=batch_size)

_, acc = model_RNN.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("\n Test accuracy: %.1f%%" % (100.0 * acc)) 