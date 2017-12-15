"""
Created on Thu Dec 15 20:32:51 2017

@author: Franjo, Dario
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
from deep_nn import VGG16
from sklearn.model_selection import train_test_split


class NeuralNetwork:
	def __init__(self):
		self.model=VGG16()
		return
	

	#as input takes list of images and path to output file(y-results)
	def train(self,listOfPics,trueOutput,normalizeInput=0):
		self.num=len(listOfPics)
		if(normalizeInput):
			meanImage,sumImage = getMeanImage(listOfPics)
			self.meanImage=meanImage
			self.sumImage=sumImage
		text_file = open(trueOutput, "r")
		self.lines = text_file.readlines()	#need to strip \n with .rstrip()
		for x_name,y in zip(listOfPics,self.lines):
			y=y.rstrip()
			x=imread(x_name)
			if(normalizeInput):
				x-=meanImage
			self.model.fit(x, y, epochs=100, batch_size=1,  verbose=1)  #nije mi jasno da li je ovo online ucenje
		return

	def classify(self,image):
		return self.model.predict(image)


"""
Function as an input gets a list of names(paths), loads them, calculates mean image in a folder and return mean and sum image
"""
def getMeanImage(listOfDirs):
	num=len(listOfDirs)
	sumImage = np.zeros(imread(listOfDirs[0]).shape)
	for i,name in enumerate(listOfDirs):
		sumImage = np.add(sumImage,imread(name))
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
	root = os.getcwd()+"/output"
	rootOutput = os.getcwd()+"/output/ssims.txt"
	listOfDirs = getFiles(root)
	listOfPics = listOfDirs[0]
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
	X_train5, X_test5, y_train5, y_test5 = train_test_split(X, y, test_size=0.33, random_state=42)

