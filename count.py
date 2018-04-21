#!/bin/python  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2    as cv
import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

from keras.datasets import cifar10

from keras.utils      import np_utils
from keras.models     import Sequential
from keras.layers     import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error

np.random.seed(123) 

##
def create_model(IMGSIZE,NUMCLASSES):
    
    model = Sequential()
    
    model.add(Convolution2D(32, (2, 2), padding='same', input_shape=IMGSIZE, activation='relu'))
    model.add(Convolution2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64, (2, 2), padding='same', activation='relu'))
    model.add(Convolution2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUMCLASSES, activation='softmax'))

    model.summary()
    
    lr = 0.01
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    
    return model

##
def load_dataset():

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = np_utils.to_categorical(y_train, np.unique(y_train).size)
    y_test  = np_utils.to_categorical(y_test , np.unique(y_test ).size)
    
    return (x_train, y_train), (x_test, y_test)

##
def plot_scores(history):
    
    fig, ax1 = plt.subplots()
    fig.suptitle('Learning Scores')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    loss,     = ax1.plot(history.history['loss'    ],'r'  ,label='Training Loss')
    val_loss, = ax1.plot(history.history['val_loss'],'r--',label='Validation Loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    acc,      = ax2.plot(history.history['acc'     ],'g'  ,label='Training Acc')
    val_acc,  = ax2.plot(history.history['val_acc' ],'g--',label='Validation Acc')
    plt.legend(handles=[loss,acc,val_loss,val_acc])
    fig.tight_layout()
    fig.savefig('plots/scores.pdf') 
    plt.show()
    
##
def draw_example(x_test,y_test):
    
    c = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']    
    i = np.random.randint(x_test.shape[0])
    print(c[y_test[i].argmax(axis=0)])    
    cv.imwrite('color_img.jpg',x_test[i])
    cv.imshow("image",x_test[i])
    cv.waitKey()
 
## Main
if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_dataset()

    IMGSIZE    = x_train.shape[1:]
    NUMCLASSES = y_train.shape[1]
    
    model = create_model(IMGSIZE,NUMCLASSES)

    batch_size = 1024
    epochs     = 1000
    
    history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test))

    plot_scores(history)
    draw_example(x_test,y_test)
    
