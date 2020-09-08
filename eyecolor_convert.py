#!/usr/bin/env python
# coding: utf-8

# In[119]:


from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from keras.initializers import RandomNormal
from matplotlib import image
import numpy as np
from PIL import Image
import cv2
import dlib
import tensorflow.compat.v1 as tf
from tensorflow import Graph

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# In[120]:


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g
def define_generator(image_shape=(256,256,3)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model


# In[121]:


##create model
image_shape = (256,256,3)


g_model = define_generator(image_shape)
model = g_model
model.load_weights('model_hong.h5')

# In[92]:


from collections import OrderedDict
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("eye_l", (36, 42)),
    ("eye_r",(42, 48))
])
colors = [(255, 0, 0)
          ,(255,0,0)] #(79, 76, 240), (230, 159, 23),
          #  (168, 100, 168), (158, 163, 32),
          #  (163, 38, 32), (180, 42, 220)]

def highligther(image, shape, colors, alpha=0.75):
    overlay = image.copy()
    output = image.copy()
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]
        hull = cv2.convexHull(pts)
        cv2.drawContours(overlay, [hull], -1, colors[i], -1)
        
        hull = cv2.convexHull(pts, returnPoints = False)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


# In[123]:


def eyecolor_make(file_name,color):
    try:
        img = cv2.imread(file_name)
        faces = detector(img)
        face = faces[0]
        dlib_shape = predictor(img,face)

        shape_2d = np.array([[p.x,p.y]for p in dlib_shape.parts()])
        eye_x_l=shape_2d[37][0]
        eye_y_l=shape_2d[37][1]-5
        eye_x2_l=shape_2d[40][0]+5
        eye_y2_l=shape_2d[40][1]

        eye_x_r=shape_2d[43][0]
        eye_y_r=shape_2d[43][1]-5
        eye_x2_r=shape_2d[46][0]+5
        eye_y2_r=shape_2d[46][1]

        eye_x_l2=shape_2d[36][0]-5
        eye_y_l2=shape_2d[37][1]-5
        eye_x2_l2=shape_2d[39][0]+5
        eye_y2_l2=shape_2d[40][1]+5

        eye_x_r2=shape_2d[42][0]-5
        eye_y_r2=shape_2d[43][1]-5
        eye_x2_r2=shape_2d[45][0]+5
        eye_y2_r2=shape_2d[46][1]+5

        face_img= highligther(img,shape_2d,colors)

        img_l = img[eye_y_l:eye_y2_l,eye_x_l:eye_x2_l]
        img_r = img[eye_y_r:eye_y2_r,eye_x_r:eye_x2_r]
        img_l2= face_img[eye_y_l:eye_y2_l,eye_x_l:eye_x2_l]
        img_r2= face_img[eye_y_r:eye_y2_r,eye_x_r:eye_x2_r]

        result = img.copy()

        target_color=color

        for i in range(img_l.shape[0]):
            for j in range(img_l.shape[1]):
                if (img_l)[:,:,0][i][j]<150 and (img_l)[:,:,1][i][j]<150 and (img_l)[:,:,2][i][j]<150 and (img_l2)[:,:,0][i][j]>150 and (img_l2)[:,:,1][i][j]<150 and (img_l2)[:,:,2][i][j]<150:
                    result[eye_y_l:eye_y2_l,eye_x_l:eye_x2_l][i][j] = target_color
        for i in range(img_r.shape[0]):
            for j in range(img_r.shape[1]):
                if (img_r)[:,:,0][i][j]<150 and (img_r)[:,:,1][i][j]<150 and (img_r)[:,:,2][i][j]<150 and (img_r2)[:,:,0][i][j]>150 and (img_r2)[:,:,1][i][j]<150 and (img_r2)[:,:,2][i][j]<150:
                    result[eye_y_r:eye_y2_r,eye_x_r:eye_x2_r][i][j] = target_color
        new_eye_l =result[eye_y_l2:eye_y2_l2,eye_x_l2:eye_x2_l2]
        new_eye_r = result[eye_y_r2:eye_y2_r2,eye_x_r2:eye_x2_r2]
        real_eye_l =img[eye_y_l2:eye_y2_l2,eye_x_l2:eye_x2_l2]
        real_eye_r = img[eye_y_r2:eye_y2_r2,eye_x_r2:eye_x2_r2]
        actual_size1 = new_eye_l.shape
        actual_size2 = new_eye_r.shape
        new_eye_l = cv2.resize(new_eye_l,(256,256))
        new_eye_r = cv2.resize(new_eye_r,(256,256))
        temp1 = np.zeros((1,256,256,3))
        temp1[0] = new_eye_l
        temp1 = (temp1 - 127.5) / 127.5

        temp2 = np.zeros((1,256,256,3))
        temp2[0] = new_eye_r
        temp2 = (temp2 - 127.5) / 127.5
        
        gen_image1 = model.predict(temp1)
        gen_image2 = model.predict(temp2)
        #plot_images(temp, temp2, gen_image)
        gen_image1 = (gen_image1+1)/2.0
        gen_image2 = (gen_image2+1)/2.0

        eye_l = cv2.resize(gen_image1[0],(actual_size1[1],actual_size1[0]))
        eye_r = cv2.resize(gen_image2[0],(actual_size2[1],actual_size2[0]))

        result = img.copy()
        result[eye_y_l2:eye_y2_l2,eye_x_l2:eye_x2_l2] = eye_l*255
        result[eye_y_r2:eye_y2_r2,eye_x_r2:eye_x2_r2] = eye_r*255
        return result
    except IndexError:
        print("dd")


