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
from deep_nn import VGG16,AntonioMax,AntonioAvg
from sklearn.model_selection import train_test_split
from keras.models import load_model


class NeuralNetwork:
	def __init__(self):
		#self.model=VGG16(include_top=True, weights=None)
		#self.model=VGG16()
		self.model=AntonioMax()
		#self.model=AntonioAvg()
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
		self.model.fit(X, Y, epochs=epochs, batch_size=batch,  verbose=1)
		return

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
    #root = "D:\FER\Neural nets\\101_ObjectCategories\\accordion"
    #root = os.getcwd()+"/output"
    #rootOutput = os.getcwd()+"/output/ssims.txt"
    #listOfDirs = getFiles(root)
    #listOfPics = listOfDirs[0]
    X = np.load("X_data.npy")
    Y = np.load("Y_data.npy")

    #meanImage,sumImage = getMeanImage(listOfPics)
    """
    num=len(listOfDirs)
    sumImage = np.zeros(imread(listOfDirs[0]).shape)
    for i,name in enumerate(listOfDirs):
        sumImage = np.add(sumImage,imread(name))
        if i%500==0:
            print(i)
    print (sumImage)
    meanImage=sumImage/num
    print (meanImage)
    """
    #X_small = X[:82300]
    #y_small2 = Y[:82300]

    X_small2 = np.load("X2.npy")
    y_small2 = np.load("Y2.npy")
    #np.save("Y2.npy",y_small2)

    y_small = y_small2[0:30000]
    X_small = X_small2[0:30000,:,:,:]
    
    X_train5, X_test5, y_train5, y_test5 = train_test_split(X_small, y_small, test_size=0.33, random_state=42)

    print(X_train5.shape, X_test5.shape, y_train5.shape, y_test5.shape)

    network = NeuralNetwork()

    #print("Training")
    #network.train(X_train5, y_train5,100)
    #print("Done training")

    network.load('my_model_Antonio.h5')
    predicted = network.classify(X_test5)
    

    print("Predicted Values:")
    print(predicted)

    predicted = np.asarray(predicted)
    predicted = np.reshape(predicted, (y_test5.shape[0], 1))

    y_test = np.asarray(y_test5)
    y_test = np.reshape(y_test, (y_test5.shape[0], 1))
    
    print("Difference")
    print(predicted - y_test)
    Difference=predicted - y_test

    np.savetxt("foo.csv", Difference, delimiter=",")

    #SAVE MODEL
    #network.save('my_model_Antonio.h5')
    
