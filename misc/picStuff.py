"""
A script designed to 1) resize all of the downloaded images to desired dimension (DEFAULT 64x64 pixels) and 2) rename images in folders from 1.png to n.png for ease of use in training
"""

import os
import scipy.misc
import random

root='./fullimages'


#Set your own PATH 
PATH = os.path.normpath('C:/Users/kenny/Desktop/toGit/misc/smallimages/')

for subdir, dirs, files in os.walk(root):
    style = subdir[2:]
    name =  style
    if len(style) < 1:
        continue
    try:
        os.stat(PATH + name)
    except:
        os.mkdir(PATH + name)
    
    i = 0
    for f in files:
        source = style + '\\' + f
        print(str(i) + source)
        try:
            image = scipy.misc.imread(source)
            image = scipy.misc.imresize(image,(64,64))
            scipy.misc.imsave(PATH + name + '\\' + str(i) + '.png',image)
            i+=1
        except Exception:
            print('missed it: ' + source)
