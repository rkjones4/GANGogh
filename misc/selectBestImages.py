"""A file that contains various methods that can be ported into the training method to select 'better' generated images"""

PATH = "." #Set your own path

#Recommended default select method, greedily selects generated images that are classified correctly according to their generated label and have a 'pretty good' realness classification score
def generate_good_images(iteration,thresh=.95):
    NUM_TO_MAKE = BATCH_SIZE
    TRIES = BATCH_SIZE*5
    CONF_THRESH = thresh
    for i in range(CLASSES):
        l = 0
        curLabel= genRandomLabels(BATCH_SIZE,CLASSES,condition=i)
        j = 0
        images = None
        while(j<NUM_TO_MAKE and l<TRIES):
            genr = Generator(BATCH_SIZE, CLASSES, sample_labels)[0]
            samples = session.run(genr, feed_dict={sample_labels: curLabel})
            samples = np.reshape(samples,[-1, 3, 64, 64])
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            prediction,accuracy = session.run([disc_real_class,realAccuracy] , feed_dict = {all_real_data_conv: samples, all_real_label_conv: curLabel})
            guess = np.argmax(prediction,1)
            my_equal = np.equal(guess,np.argmax(curLabel,1))
            for s,_ in enumerate(prediction):
                prediction[s] = prediction[s]/np.sum(prediction[s])
                confidence = np.amax(prediction,1)
                for k,image in enumerate(samples):
                    if guess[k] == i and confidence[k]>CONF_THRESH and j < NUM_TO_MAKE:
                        if isinstance(images, np.ndarray):
                            images = np.concatenate((images,image),0)
                        else:
                            images = image
                    j+=1
                l += 1
            CONF_THRESH = CONF_THRESH * .9
        try:
            samples = images
            lib.save_images.save_images(samples.reshape((-1, 3, 64, 64)), PATH + '/good_samples_{}.png'.format(str(i)))
        except Exception as e:
            print(e)

#More intensive method used to generative most evocative results, out of a series of generated batches of images ranks all correctly classified images according to realness value and degree of condifence in the classification, only returns images that do the best in both metrics, can take awhile
def generate_best_images():
        LOOK_AT = 10
        RETUR = 64
        test = [6,4,10]
        for i in range(CLASSES):
            print(i)
            curLabel= genRandomLabels(BATCH_SIZE,CLASSES,test[i])

            images = None
            thoughts = []
            index = 0
            for j in range(LOOK_AT):
                genr = Generator(BATCH_SIZE, CLASSES, sample_labels)[0]
                samples = session.run(genr, feed_dict={sample_labels: curLabel})
                samples = np.reshape(samples,[-1, 3, DIMI, DIMI])
                samples = ((samples+1.)*(255.99/2)).astype('int32')
                
                prediction,accuracy,realness = session.run([disc_real_class,realAccuracy,disc_real] , feed_dict = {all_real_data_conv: samples, all_real_label_conv: curLabel})
                
                guess = np.argmax(prediction,1)
                my_equal = np.equal(guess,np.argmax(curLabel,1))
                prediction = prediction.clip(min=.001)
                for s,_ in enumerate(prediction):
                    prediction[s] = prediction[s]/np.sum(prediction[s])
                confidence = np.amax(prediction,1)
                for k,image in enumerate(samples):
                    if guess[k] == i:
                        if isinstance(images, np.ndarray):
                            images = np.concatenate((images,[image]),0)
                        else:
                            images = np.array([image])
                        thoughts.append([confidence[k],realness[k],index])
                        index += 1

            thoughts.sort(key=lambda x: x[0])
            
            thoughts.reverse()
            thoughts = thoughts[:3*64]
            thoughts.sort(key=lambda x: x[1])
            thoughts.reverse()
            thoughts = thoughts[:RETUR]
            
            indexBase = []
            for t in thoughts:
                indexBase.append(t[2])
            print(indexBase)
                
            samples = None
            try:
                for k,image in enumerate(images):
                    if k in indexBase:
                        if isinstance(samples,np.ndarray):
                            samples = np.concatenate((samples,image),0)
                        else:
                            samples = image
                lib.save_images.save_images(samples.reshape((-1, 3, DIMI, DIMI)), PATH + '/best_samples_{}.png'.format(str(i)))
            except Exception as e:
                print(e)
 

