"""
Created on Thu Dec 15 20:32:51 2017

@author: Franjo, Dario, Matej
"""
#https://www.datacamp.com/community/blog/keras-cheat-sheet
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from __future__ import print_function
import skimage
import numpy as np, os
from skimage.util import random_noise
from skimage.viewer.utils import new_plot
from skimage.transform import resize
from skimage.io import imread_collection,imshow,imsave,imread
import subprocess
import csv
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from deep_nn_ala_v2 import VGG16
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self):
        #self.model=VGG16(include_top=True, weights=None)
        self.model=VGG16()
        return


    #as input takes list of images and path to output file(y-results)
    def train(self,X,Y,epochs=10,batch=64,normalizeInput=0):
        print(epochs)
        self.num=len(X)
        #if(normalizeInput):
            #    meanImage,sumImage = getMeanImage(X)
            #    self.meanImage=meanImage
            #    self.sumImage=sumImage
            #for x,y in zip(X,Y):
            #   if(normalizeInput):
            #      x-=meanImage
        self.model.summary()
                
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch, validation_split=0.33, verbose=1)
    

    def classify(self,image):
        return self.model.predict(image)

    def save(self,filename):
        self.model.save(filename)

    def load(self,filename):
        self.model = load_model(filename)


"""
Function as an input gets a list of names(paths), loads them, calculates mean image in a folder and return mean and sum image
"""

def getMeanImage(X):
    num=len(X)
    sumImage = np.zeros(X[0].shape)
    for i,pic in enumerate(X):
        sumImage = np.add(sumImage,pic)
        if i%500==0:
            print(i)
    #print (sumImage)
    meanImage=sumImage/num
    #print (meanImage)
    return meanImage,sumImage


def getFiles(root):
    print(root)
    listOfNames=[]
    for path, subdirs, files in os.walk(root):
        #print subdirs
        #print(files)
        for name in files:
            if(len(name.split("."))>1):
                if(name.split(".")[1]=="jpg"):
                    #print(os.path.join(path, name))
                    listOfNames.append(os.path.join(path, name))
                    #print (os.path.join(path, name))
    #print (listOfNames)
    Collection = imread_collection(listOfNames)
    #imshow(Collection[0])
    #imshow(random_noise(Collection[0]))
    return listOfNames,Collection
    pass


if __name__ == "__main__":
    
    # PREIMENOVATI ZA UCITAVANJE <<<<<<<<<<<<<<
    X_small = np.load("X_data.npy")
    y_small = np.load("Y_data.npy")
    print("Before:")
    
    print(X_small.shape)
    X_small = np.expand_dims(X_small, axis=-1)    
    print("After")
    print(X_small.shape)
    
    network = NeuralNetwork()

    print("Training")
    history = network.train(X_small, y_small, 150)#<<<<<<<<<<<<
    print("Done training")
    predicted = network.classify(X_small)
    print("Predicted Values:")
    print(predicted)
    predicted = np.asarray(predicted)
    predicted = np.reshape(predicted, (y_small.shape[0], 1))
    y_test = np.asarray(y_small)
    y_test = np.reshape(y_test, (y_small.shape[0], 1))
    
    difference=np.abs(predicted - y_test)
    print(predicted.shape,y_test.shape,difference.shape)
    np.savetxt("foo.csv", np.hstack((y_test,predicted,difference)), delimiter=",")
    network.save('my_model_Antonio.h5')
    
        
    # list all data in history
    print(history.history.keys())
    fig = plt.figure()
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig('accuracy.png')
    
    
    fig2 = plt.figure()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig2.savefig('MSE.png')
    
    
    
    
    
    
