# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:32:51 2017

@author: Franjo
"""

"""
POVEZNICA ZA SSIM ALGORITAM IMPLEMENTIRAN U PYTHONU
https://github.com/jterrace/pyssim
"""

import skimage
import numpy as np, os
from skimage.util import random_noise
from skimage.viewer.utils import new_plot
from skimage.transform import resize
from skimage.io import imread_collection,imshow,imsave
import subprocess
import csv
#import imghdr

listOfAdds = ['','gaussian','poisson','salt','pepper','s&p','speckle','gaussian&s&p']

def getDirectories(root):
    listOf=[]
    listOfDirs=[]
    for path, subdirs, files in os.walk(root):
        listOf.append([path, subdirs, files])
    for member in listOf[1:]:
        listOfDirs.append(member[0])
    return listOfDirs

def getFiles(root):
    print(root)
    listOfNames=[]
    for path, subdirs, files in os.walk(root):
        #print subdirs
        #print(files)
        for name in files:
            if(len(name.split("."))>1):
                if(name.split(".")[1]=="jpg"):
                    listOfNames.append(os.path.join(path, name))
                #print (os.path.join(path, name))
    #print (listOfNames)
    Collection = imread_collection(listOfNames)
    #imshow(Collection[0])
    #imshow(random_noise(Collection[0]))
    return listOfNames,Collection
    pass    

def addNoise(Collection):
    listOfLists = []
    i=0
    for picture in Collection:
        picture=resize(picture,(256,256))
        gaussian = random_noise(picture)
        poisson = random_noise(picture,mode='poisson')
        salt = random_noise(picture,mode='salt')
        pepper = random_noise(picture,mode='pepper')
        sp = random_noise(picture,mode='s&p')
        speckle = random_noise(picture,mode='speckle')
        mix = random_noise(sp)
        listOfLists.append([picture,gaussian,poisson,salt,pepper,sp,speckle,mix])
        print (i)
        i+=1
    return listOfLists


def saveFiles(listOfNames,listOfLists):
    saveNames=[]
    for name,pictures in zip(listOfNames,listOfLists):
        saveName=[]
        if(not os.path.exists(name.split(".")[0])):
            os.makedirs(name.split(".")[0])
        for helpName,picture in zip(listOfAdds,pictures):
            add=(name.split(".")[0]).split("\\")[-1]
            imsave(name.split(".")[0]+"\\"+add+"_"+helpName+'.jpg',picture)
            saveName.append(name.split(".")[0]+"\\"+add+"_"+helpName+'.jpg')
        saveNames.append(saveName)
    return saveNames
            
    pass

def calculateSSIM(listOfLists):
    listOfSSIM = []
    for Names in listOfLists:
        list_SSIM = []
        for i in range(len(Names)):
            p1 = subprocess.Popen(['pyssim', Names[0],Names[i]], stdout=subprocess.PIPE,shell=True)
            output = p1.communicate()[0]
            list_SSIM.append(output)
        listOfSSIM.append(list_SSIM)
    return listOfSSIM

def saveSSIM(result,saveNames,saveTo):
    with open(saveTo+"\\"+"test.txt","a+") as csvfile:
        #spamwriter = csv.writer(csvfile, delimiter=' ')
        #csvfile.write(listOfAdds)
        for res,name in zip(result,saveNames):
            for i,j in zip(res,name):
                csvfile.write(j+" "+i.rstrip()+"\n")
                #spamwriter.writerow(name+res)
    pass

if __name__ == "__main__":
    #root = "D:\FER\Neural nets\\101_ObjectCategories\\accordion"
    root = "D:\FER\Neural nets\\101_ObjectCategories"
    listOfDirs = getDirectories(root)
    for _dir in listOfDirs:
        listOfNames,Collection=getFiles(_dir)
        listOfLists=addNoise(Collection)
        names=saveFiles(listOfNames,listOfLists)
        result=calculateSSIM(names)
        saveSSIM(result,names,_dir)
    pass