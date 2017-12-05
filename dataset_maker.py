import os
from skimage.measure import compare_ssim 
from skimage.util import random_noise
from skimage.transform import resize
from skimage.io import imread, imsave

def addNoise(picture):
    noised = []
    noised.append(random_noise(picture))
    noised.append(random_noise(picture,mode='poisson'))
    noised.append(random_noise(picture,mode='gaussian'))
    noised.append(random_noise(picture,mode='salt'))
    noised.append(random_noise(picture,mode='pepper'))
    noised.append(random_noise(picture,mode='speckle'))
    sp = random_noise(picture,mode='s&p')
    noised.append(sp)
    noised.append(random_noise(sp))
    return noised

file_Y = open("./output/ssims.txt", "w")
num = 0
for root, dirs, files in os.walk('/home/dsitnik/Faks/Neuronske mreže/TestImgs/'):
    for file in files:
        if num % 90 == 0:
            print(num)
        picture = imread(os.path.abspath(os.path.join(root, file)), as_grey=True)
        picture = resize(picture,(128,128))
        noises = addNoise(picture)
        noises.append(picture)
        for image in noises:
            file_Y.write(str(compare_ssim(image, picture))+"\n")
            imsave("./output/" + str(num).zfill(6)+".jpg", image)
            num+=1
file_Y.close()
