""" Creates batches of images to feed into the training network conditioned by genre, uses upsampling when creating batches to account for uneven distributuions """


import numpy as np
import scipy.misc
import time
import random
import os
#Set the dimension of images you want to be passed in to the network
DIM = 64

#Set your own path to images
path = os.path.normpath('C:/Users/kenny/Desktop/toGit/misc/smallimages/')

#This dictionary should be updated to hold the absolute number of images associated with each genre used during training
styles = {'abstract': 14794,
          'animal-painting': 1319,
          'cityscape': 5833,
          'figurative': 3335,
          'flower-painting': 1260,
          'genre-painting': 14881,
          'landscape': 14893,
          'marina': 1199,
          'mythological-painting': 1670,
          'nude-painting-nu': 2276,
          'portrait': 14496,
          'religious-painting': 7915,
          'still-life': 2314,
          'symbolic-painting': 2454}

styleNum = {'abstract': 0,
            'animal-painting': 1,
            'cityscape': 2,
            'figurative': 3,
            'flower-painting': 4,
            'genre-painting': 5,
            'landscape': 6,
            'marina': 7,
            'mythological-painting': 8,
            'nude-painting-nu': 9,
            'portrait': 10,
            'religious-painting': 11,
            'still-life': 12,
            'symbolic-painting': 13}

curPos = {'abstract': 0,
          'animal-painting': 0,
          'cityscape': 0,
          'figurative': 0,
          'flower-painting': 0,
          'genre-painting': 0,
          'landscape': 0,
          'marina': 0,
          'mythological-painting': 0,
          'nude-painting-nu': 0,
          'portrait': 0,
          'religious-painting': 0,
          'still-life': 0,
          'symbolic-painting': 0}

testNums = {}
trainNums = {}

#Generate test set of images made up of 1/20 of the images (per genre)
for k,v in styles.items():
    # put a twentieth of paintings in here
    nums = range(v)
    random.shuffle(list(nums))
    testNums[k] = nums[0:v//20]
    trainNums[k] = nums[v//20:]

def inf_gen(gen):
    while True:
        for (images,labels) in gen():
            yield images,labels
            
    

def make_generator(files, batch_size, n_classes):
    if batch_size % n_classes != 0:
        raise ValueError("batch size must be divisible by num classes")

    class_batch = batch_size // n_classes

    generators = []
    
    def get_epoch():

        while True:

            images = np.zeros((batch_size, 3, DIM, DIM), dtype='int32')
            labels = np.zeros((batch_size, n_classes))
            n=0
            for style in styles:
                styleLabel = styleNum[style]
                curr = curPos[style]
                for i in range(class_batch):
                    if curr == styles[style]:
                        curr = 0
                        random.shuffle(list(files[style]))
                    t0=time.time()
                    image = scipy.misc.imread("{}/{}/{}.png".format(path, style, str(curr)),mode='RGB')
                    #image = scipy.misc.imresize(image,(DIM,DIM))
                    images[n % batch_size] = image.transpose(2,0,1)
                    labels[n % batch_size, int(styleLabel)] = 1
                    n+=1
                    curr += 1
                curPos[style]=curr

            #randomize things but keep relationship between a conditioning vector and its associated image
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
            yield (images, labels)
                        

        
    return get_epoch

def load(batch_size):
    return (
        make_generator(trainNums, batch_size, 14),
        make_generator(testNums, batch_size, 14),
    )

#Testing code to validate that the logic in generating batches is working properly and quickly
if __name__ == '__main__':
    train_gen, valid_gen = load(100)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        a,b = batch
        print(str(time.time() - t0))
        if i == 1000:
            break
        t0 = time.time()
