import numpy as np
import sys
import os
from scipy.misc import imread, imshow, imresize, imsave
import tensorflow as tf
from scipy import ndimage

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=200)

#input files
images_dir = '../Data/images/'
labels_dir = '../Data/labels/'

#prefix in name of input file
nam = 'out'


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)

##max pooling layer
# Input : x
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

normalizer_fn = None
normalizer_fn = tf.contrib.layers.batch_norm

## Convolution layer with batch normalization
# Input of previous layer : x, 
# Number of outputs of this layer : F
# Kernel size : k X k
def con_bn(x, k, F):
    return tf.contrib.layers.convolution2d(inputs=x,
                                           num_outputs=F,
                                           kernel_size=[k, k],
                                           stride=[1, 1],
                                           padding='SAME',
                                           # activation_fn=None,
                                           activation_fn=tf.nn.relu,
                                           normalizer_fn=normalizer_fn,
                                           normalizer_params=None,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                           biases_initializer=tf.zeros_initializer(),
                                           trainable=True)


## Fully connected layer with batch normalization
# Input of previous layer : x, 
# Number of outputs of this layer : F
def fc_bn(x, F):
    return tf.contrib.layers.fully_connected(inputs=x,
                                             num_outputs=F,
                                             activation_fn=tf.nn.relu,
                                             normalizer_fn=normalizer_fn,
                                             normalizer_params=None,
                                             weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                             biases_initializer=tf.zeros_initializer(),
                                             trainable=True)


## Fully connected layer without batch normalization
# Input of previous layer : x, 
# Number of outputs of this layer : F

def fc(x, F):
    return tf.contrib.layers.fully_connected(inputs=x,
                                             num_outputs=F,
                                             activation_fn=None,
                                             # activation_fn=tf.nn.relu,
                                             normalizer_fn=None,
                                             normalizer_params=None,
                                             weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32),
                                             biases_initializer=tf.zeros_initializer(),
                                             trainable=True)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)


# Fraction to which image is downsampled : PER 
# Dimensions of image to be passed to be passed to CNN : SZ1 X SZ2 
PER = .5
SZ1 = 576
SZ2 = 324

# Number of output parameters : 48 for the curling coefficients
number_out = 48

x = tf.placeholder(tf.float32, [None, SZ1, SZ2, 1])

y_ = tf.placeholder(tf.float32, [None, number_out])


# 2 convolution layers followed by a max pooling layer; 5 times the same thing

first_conv = 8

h_conv11 = con_bn(x, 5, first_conv)
h_conv12 = con_bn(h_conv11, 5, first_conv)
h_pool1 = max_pool_2x2(h_conv12)

second_conv = 2*first_conv

h_conv21 = con_bn(h_pool1, 3, second_conv)
h_conv22 = con_bn(h_conv21, 3, second_conv)
h_pool2 = max_pool_2x2(h_conv22)

third_conv = 2*second_conv

h_conv31 = con_bn(h_pool2, 3, third_conv)
h_conv32 = con_bn(h_conv31, 3, third_conv)
h_pool3 = max_pool_2x2(h_conv32)

fourth_conv = 2*third_conv

h_conv41 = con_bn(h_pool3, 3, fourth_conv)
h_conv42 = con_bn(h_conv41, 3, fourth_conv)
h_pool4 = max_pool_2x2(h_conv42)

fifth_conv = fourth_conv

h_conv51 = con_bn(h_pool4, 3, fifth_conv)
h_conv52 = con_bn(h_conv51, 3, fifth_conv)
h_pool5 = max_pool_2x2(h_conv52)


# Output dimension of penultimate layer : fc_size
fc_size = 264*48 

#flatten 2d output to 1d 
h_pool5_flat = tf.reshape(h_pool5, [-1, fc_size])

# final output
y_conv = fc(h_pool5_flat, number_out)

# Weighted outputs to be passed to error function
output_weights = tf.constant([2.0,1.0,4.0,5.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,4.0,1.0,1.0,3.0,5.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,4.0,2.0,1.0,3.0,5.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,1.0,1.0,3.0,5.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,4.0])

error = tf.reduce_mean((tf.square(tf.multiply(tf.subtract(y_conv, y_),output_weights))))
train_step = tf.train.AdamOptimizer(1e-4).minimize(error)   

#Number of images used in training 
train_data_size = 35000
train_data_image = np.zeros(shape=(train_data_size, SZ1, SZ2, 1), dtype='float32')
train_data_label = np.zeros(shape=(train_data_size, 48), dtype='float32')


#read data
ind1 = 0
while ind1 < train_data_size:
    if(ind1 % 100) == 0:
        print(ind1, train_data_size)
    train_data_image[ind1][:, :, 0] = imresize(imread(os.path.join(images_dir, nam + str(ind1+1) + '.png'), mode='L'), PER)
    train_data_label[ind1] = np.reshape(np.genfromtxt(os.path.join(labels_dir, 'tmp' + str(ind1+1) + '.txt')), -1)
    ind1 = ind1 + 1

valid_data_size = 1000
valid_data_image = np.zeros(shape=(valid_data_size, SZ1, SZ2, 1), dtype='float32')
valid_data_label = np.zeros(shape=(valid_data_size, 48), dtype='float32')

ind2 = 0
while ind2 < valid_data_size:
    valid_data_image[ind2][:, :, 0] = imresize(imread(os.path.join(images_dir, nam + str(ind1+1) + '.png'), mode='L'), PER)
    valid_data_label[ind2] = np.reshape(np.genfromtxt(os.path.join(labels_dir, 'tmp' + str(ind1+1) + '.txt')), -1)
    ind1 = ind1 + 1
    ind2 = ind2 + 1


init_op = tf.global_variables_initializer()

sess.run(init_op)

sum_los = -1.0
sum_losTrain = -1.0

j = 0

BATCH_SIZE = 32
VALID_BATCH_SIZE = 32

#save output in this file
f = open('output.txt','w')

# loop size : number of training iterations
for i in range(10000):
    #reset counter after almost all the training data set is covered
    if (i % (train_data_size/(BATCH_SIZE+1) -1 ) ) == 0:
        j = 0
    #Get validation error and training error after every 50th step
    if (i % 50 ) == 0:
        valid_data_image, valid_data_label = shuffle_in_unison(valid_data_image, valid_data_label)
        print("length of input : ",len(train_data_image))
        losTrain = error.eval(feed_dict={x: train_data_image[0:BATCH_SIZE], y_: train_data_label[0:BATCH_SIZE]})
        losValid = error.eval(feed_dict={x: valid_data_image[0:VALID_BATCH_SIZE], y_: valid_data_label[0:VALID_BATCH_SIZE]})
        y2 = y_conv.eval(feed_dict={x: valid_data_image[0:VALID_BATCH_SIZE]})#, y_: valid_data_label[0:VALID_BATCH_SIZE]})
        if sum_los < 0:
            sum_los = losValid
        fraction=0.7
        sum_los = sum_los*(fraction) + losValid*(1-fraction)
        if sum_losTrain < 0:
            sum_losTrain = losTrain
        sum_losTrain = sum_losTrain*(fraction) + losTrain*(1-fraction)
        print("sum loss valid : ",sum_los, " loss valid : ", losValid," sum_loss train : ", sum_losTrain, " loss2 train : ", losTrain," i : ",  i)
        f.write("sum loss valid : "+ str(sum_los)+ " loss valid : "+ str(losValid)+" sum_loss train : "+ str(sum_losTrain)+ " loss2 train : "+ str(losTrain)+" i : "+  str(i) + '\n')
        print( " y2i : ", y2[0])
        f.write(" y2i : "+ str( y2[0]))
        print(" valid : ", valid_data_label[0])
        f.write(" valid : "+ str(valid_data_label[0]))

        print("y2 shape : ", y2.shape, " , valid data label shape : ", valid_data_label[0:VALID_BATCH_SIZE].shape)
        component_error = np.mean(np.absolute(y2-valid_data_label[0:VALID_BATCH_SIZE]), axis=0)
        print("component error ", component_error)

    j = j+1
    #training step
    train_step.run(feed_dict={x: train_data_image[(j*BATCH_SIZE):((j+1)*BATCH_SIZE)], y_: train_data_label[(j*BATCH_SIZE):((j+1)*BATCH_SIZE)]})
f.close()
