import os
from skimage.measure import structural_similarity as ssim
from skimage.util import random_noise
from skimage.viewer.utils import new_plot
from skimage.transform import resize
from skimage.io import imread,imshow,imsave

def addNoise(picture):
    noised = []
    noised.append(random_noise(picture))
    noised.append(random_noise(picture,mode='poisson'))
    noised.append(random_noise(picture,mode='salt'))
    noised.append(random_noise(picture,mode='pepper'))
    sp = random_noise(picture,mode='s&p')
    noised.append(sp)
    noised.append(random_noise(picture,mode='speckle'))
    noised.append(random_noise(sp))
    return noised

file_Y = open("ssims.txt", "w")
num = 0
for root, dirs, files in os.walk('./'):
    for file in files:
        picture = imread(file, as_grey=True)
        picture = resize(file,(128,128))
        noises = addNoise(picture)
        noises.append(picture)
        for image in noises:
            file_Y.write(str(ssim(image, picture))+"\n")
            imsave("./output/"+ str(num).zfill(6)+".jpg", image)
            num+=1
file_Y.close()
