import os, sys
sys.path.append(os.getcwd())

from random import randint

import time
import functools
import math

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.wikiartGenre
import tflib.ops.layernorm
import tflib.plot


MODE = 'acwgan' # dcgan, wgan, wgan-gp, lsgan
DIM = 64 # Model dimensionality
CRITIC_ITERS = 5 # How many iterations to train the critic for
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 84 # Batch size. Must be a multiple of CLASSES and N_GPUS
ITERS = 200000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64*64*3 # Number of pixels in each iamge
CLASSES = 14 #Number of classes, for genres probably 14
PREITERATIONS = 2000 #Number of preiteration training cycles to run
lib.print_model_settings(locals().copy())

def GeneratorAndDiscriminator():
    return kACGANGenerator, kACGANDiscriminator


DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Batchnorm(name, axes, inputs):
    
    if ('Discriminator' in name) and (MODE == 'wgan-gp' or MODE == 'acwgan'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(name, output_dim, a, b, c=None, d=None):
    if c is not None and d is not None:
        a = a + c
        b = b + d
        
    result = tf.sigmoid(a) * tf.tanh(b)
    return result

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2, output_dim=output_dim//2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim//2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim//2, output_dim=output_dim//2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim//2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2,  output_dim=output_dim//2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
    output = Batchnorm(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)

# ! Generators

def kACGANGenerator(n_samples, numClasses, labels, noise=None, dim=DIM, bn=True, nonlinearity=tf.nn.relu, condition=None):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    labels = tf.cast(labels, tf.float32)        
    noise = tf.concat([noise, labels], 1)

    output = lib.ops.linear.Linear('Generator.Input', 128+numClasses, 8*4*4*dim*2, noise) #probs need to recalculate dimensions
    output = tf.reshape(output, [-1, 8*dim*2, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond1', numClasses, 8*4*4*dim*2, labels,biases=False)
    condition = tf.reshape(condition, [-1, 8*dim*2, 4, 4])
    output = pixcnn_gated_nonlinearity('Generator.nl1', 8*dim, output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])


    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond2', numClasses, 4*8*8*dim*2, labels)
    condition = tf.reshape(condition, [-1, 4*dim*2, 8, 8])
    output = pixcnn_gated_nonlinearity('Generator.nl2', 4*dim,output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])
    
    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond3', numClasses, 2*16*16*dim*2, labels)
    condition = tf.reshape(condition, [-1, 2*dim*2, 16, 16])
    output = pixcnn_gated_nonlinearity('Generator.nl3', 2*dim,output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])
    
    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond4', numClasses, 32*32*dim*2, labels)
    condition = tf.reshape(condition, [-1, dim*2, 32, 32])
    output = pixcnn_gated_nonlinearity('Generator.nl4', dim, output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)

    output = tf.tanh(output)
    
    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, OUTPUT_DIM]), labels

def kACGANDiscriminator(inputs, numClasses, dim=DIM, bn=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)
    
    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    
    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = nonlinearity(output)
    finalLayer = tf.reshape(output, [-1, 4*4*8*dim])

    sourceOutput = lib.ops.linear.Linear('Discriminator.sourceOutput', 4*4*8*dim, 1, finalLayer)
    
    classOutput = lib.ops.linear.Linear('Discriminator.classOutput', 4*4*8*dim, numClasses, finalLayer)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()



    return (tf.reshape(sourceOutput, [-1]), tf.reshape(classOutput, [-1, numClasses]))

                                
def genRandomLabels(n_samples, numClasses,condition=None):
    labels = np.zeros([BATCH_SIZE,CLASSES], dtype=np.float32)
    for i in range(n_samples):
        if condition is not None:
            labelNum = condition
        else:
            labelNum = randint(0, numClasses-1)
        labels[i, labelNum] = 1
    return labels

Generator, Discriminator = GeneratorAndDiscriminator()
            
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    all_real_label_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE,CLASSES])
    
    generated_labels_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE,CLASSES])
    sample_labels_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE,CLASSES])
    
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
        split_real_label_conv = tf.split(all_real_label_conv, len(DEVICES))
        split_generated_labels_conv = tf.split(generated_labels_conv, len(DEVICES))
        split_sample_labels_conv = tf.split(sample_labels_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
        split_real_data_label = tf.split(0, len(DEVICES), all_real_data_conv)
        split_generated_labels = tf.split(0, len(DEVICES), generated_labels_conv)
        split_sample_labels = tf.split(0, len(DEVICES), sample_labels_conv)

    gen_costs, disc_costs = [],[]

    for device_index, (device, real_data_conv, real_label_conv) in enumerate(zip(DEVICES, split_real_data_conv, split_real_label_conv)):
        with tf.device(device):
            
            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE//len(DEVICES), OUTPUT_DIM])
            real_labels = tf.reshape(real_label_conv, [BATCH_SIZE//len(DEVICES), CLASSES])

            generated_labels = tf.reshape(split_generated_labels_conv, [BATCH_SIZE//len(DEVICES), CLASSES])
            sample_labels = tf.reshape(split_sample_labels_conv, [BATCH_SIZE//len(DEVICES), CLASSES])
                        
            fake_data, fake_labels= Generator(BATCH_SIZE//len(DEVICES), CLASSES, generated_labels)
            
            #set up discrimnator results
            
            disc_fake,disc_fake_class = Discriminator(fake_data, CLASSES)
            disc_real,disc_real_class = Discriminator(real_data, CLASSES)
                
            prediction = tf.argmax(disc_fake_class, 1)
            correct_answer = tf.argmax(fake_labels, 1)
            equality = tf.equal(prediction, correct_answer)
            genAccuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
            
            prediction = tf.argmax(disc_real_class, 1)
            correct_answer = tf.argmax(real_labels, 1)
            equality = tf.equal(prediction, correct_answer)
            realAccuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

            gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            gen_cost_test = -tf.reduce_mean(disc_fake)
            disc_cost_test = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
                                                                                     
            generated_class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_fake_class,
                                                                                              labels=fake_labels))
            

            real_class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_real_class,
                                                                                              labels=real_labels))
            gen_cost += generated_class_cost
            disc_cost += real_class_cost
                
            alpha = tf.random_uniform(
                shape=[BATCH_SIZE//len(DEVICES),1], 
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates, CLASSES)[0], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_cost += LAMBDA*gradient_penalty
            
            real_class_cost_gradient = real_class_cost*50 + LAMBDA*gradient_penalty
            

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)
            
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                             var_list=lib.params_with_name('Generator'),
                                                                                             colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                              var_list=lib.params_with_name('Discriminator.'),
                                                                                              colocate_gradients_with_ops=True)
    class_train_op =  tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(real_class_cost_gradient,
                                                                                                var_list=lib.params_with_name('Discriminator.'),
                                                                                                colocate_gradients_with_ops=True)
    # For generating samples
    
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = BATCH_SIZE // len(DEVICES)
        all_fixed_noise_samples.append(Generator(n_samples, CLASSES, sample_labels,noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples])[0])
        if tf.__version__.startswith('1.'):
            all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
        else:
            all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)
    
    
    def generate_image(iteration):
        for i in range(CLASSES):
            curLabel= genRandomLabels(BATCH_SIZE,CLASSES,condition=i)
            samples = session.run(all_fixed_noise_samples, feed_dict={sample_labels: curLabel})
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), 'generated/samples_{}_{}.png'.format(str(i), iteration))
    
    
    
    # Dataset iterator
    train_gen, dev_gen = lib.wikiartGenre.load(BATCH_SIZE)

    def softmax_cross_entropy(logit, y):
        return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))
    
    def inf_train_gen():
        while True:
            for (images,labels) in train_gen():
                yield images,labels


    _sample_labels = genRandomLabels(BATCH_SIZE, CLASSES)
    # Save a batch of ground-truth samples
    _x,_y = next(train_gen())
    _x_r = session.run(real_data, feed_dict={all_real_data_conv: _x})
    _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
    lib.save_images.save_images(_x_r.reshape((BATCH_SIZE, 3, 64, 64)), 'generated/samples_groundtruth.png')



    session.run(tf.initialize_all_variables(), feed_dict={generated_labels_conv: genRandomLabels(BATCH_SIZE,CLASSES)})
    gen = train_gen()
    
    for iterp in range(PREITERATIONS):
        _data, _labels = next(gen)
        _ , accuracy = session.run([disc_train_op, realAccuracy],feed_dict = {all_real_data_conv: _data, all_real_label_conv: _labels, generated_labels_conv: genRandomLabels(BATCH_SIZE, CLASSES)})
        if iterp % 100 == 99:
            print('pretraining accuracy: ' + str(accuracy))
    
            
    for iteration in range(ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op, feed_dict={generated_labels_conv: genRandomLabels(BATCH_SIZE,CLASSES)})
        # Train critic
        disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data, _labels = next(gen)
            _disc_cost, _disc_cost_test, class_cost_test, gen_class_cost, _gen_cost_test, _genAccuracy, _realAccuracy, _ = session.run([disc_cost, disc_cost_test, real_class_cost, generated_class_cost, gen_cost_test, genAccuracy, realAccuracy, disc_train_op], feed_dict={all_real_data_conv: _data, all_real_label_conv: _labels, generated_labels_conv: genRandomLabels(BATCH_SIZE,CLASSES)})
         
        lib.plot.plot('train disc cost', _disc_cost)   
        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('wgan train disc cost', _disc_cost_test)
        lib.plot.plot('train class cost', class_cost_test)
        lib.plot.plot('generated class cost', gen_class_cost)
        lib.plot.plot('gen cost cost', _gen_cost_test)
        lib.plot.plot('gen accuracy', _genAccuracy)
        lib.plot.plot('real accuracy', _realAccuracy)        
        
        if (iteration % 100 == 99 and iteration<1000) or iteration % 1000 == 999 :
            t = time.time()
            dev_disc_costs = []
            images, labels = next(dev_gen())
            _dev_disc_cost, _dev_disc_cost_test, _class_cost_test, _gen_class_cost, _dev_gen_cost_test, _dev_genAccuracy, _dev_realAccuracy = session.run([disc_cost, disc_cost_test, real_class_cost, generated_class_cost, gen_cost_test, genAccuracy, realAccuracy], feed_dict={all_real_data_conv: images, all_real_label_conv: labels, generated_labels_conv: genRandomLabels(BATCH_SIZE,CLASSES)})
            dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            lib.plot.plot('wgan dev disc cost', _dev_disc_cost_test)
            lib.plot.plot('dev class cost', _class_cost_test)
            lib.plot.plot('dev generated class cost', _gen_class_cost)
            lib.plot.plot('dev gen  cost', _dev_gen_cost_test)
            lib.plot.plot('dev gen accuracy', _dev_genAccuracy)
            lib.plot.plot('dev real accuracy', _dev_realAccuracy)        


        if iteration % 1000 == 999:
            generate_image(iteration)
            #Can add generate_good_images method in here if desired
            
        if (iteration < 10) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()

