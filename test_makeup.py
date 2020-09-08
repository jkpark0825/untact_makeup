#!/usr/bin/env python
# coding: utf-8

# In[13]:


import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import time
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
predictor = sp


def align_faces(img):
    dets = detector(img, 1)
    
    objs = dlib.full_object_detections()

    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
        
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
    
    return faces
sess = tf.Session()
#sess.run(tf.global_variables_initializer())
tf.disable_eager_execution()
saver = tf.train.import_meta_graph('models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('models'))
graph = tf.get_default_graph()

X = graph.get_tensor_by_name('X:0') # source
Y = graph.get_tensor_by_name('Y:0') # reference
Xs = graph.get_tensor_by_name('generator/xs:0') # output
def preprocess(img):
    return img.astype(np.float32) / 127.5 - 1.

def postprocess(img):
    return ((img + 1.) * 127.5).astype(np.uint8)
def tracker(src_img):
    faces = detector(src_img,1)
    face = faces[0]
    dlib_shape = predictor(src_img,face)
    shape_2d = np.array([[p.x,p.y]for p in dlib_shape.parts()])
    lip_x=shape_2d[48][0]-10
    lip_y=shape_2d[50][1]-10
    lip_x2=shape_2d[54][0]+10
    lip_y2=shape_2d[57][1]+10

    eye_x_l=shape_2d[36][0]-15
    eye_y_l=shape_2d[38][1]-15
    eye_x2_l=shape_2d[39][0]+15
    eye_y2_l=shape_2d[41][1]+15

    eye_x_r=shape_2d[42][0]-15
    eye_y_r=shape_2d[44][1]-15
    eye_x2_r=shape_2d[45][0]+15
    eye_y2_r=shape_2d[46][1]+15
    return lip_x,lip_y,lip_x2,lip_y2,eye_x_l,eye_y_l,eye_x2_l,eye_y2_l,eye_x_r,eye_y_r,eye_x2_r,eye_y2_r
def make_img(src_img,skin_img=None,lip_img = None, eye_img = None):
    lip_x,lip_y,lip_x2,lip_y2,eye_x_l,eye_y_l,eye_x2_l,eye_y2_l,eye_x_r,eye_y_r,eye_x2_r,eye_y2_r = tracker(src_img)
    
    if skin_img is None:
        img = src_img.copy()
    else: 
        img = skin_img[:,:,::-1].copy()
        img = align_faces(img)
        img = img[0]

    #img의 좌표 찾기   
    
    if lip_img is None:
        if skin_img is None:
            img = img.copy()
            #img = src_img.copy()
        else:
            lip2_x,lip2_y,lip2_x2,lip2_y2,eye2_x_l,eye2_y_l,eye2_x2_l,eye2_y2_l,eye2_x_r,eye2_y_r,eye2_x2_r,eye2_y2_r = tracker(img)
            img[lip2_y:lip2_y2,lip2_x:lip2_x2] = cv2.resize(src_img[lip_y:lip_y2,lip_x:lip_x2],(int(img[lip2_y:lip2_y2,lip2_x:lip2_x2].shape[1]),int(img[lip2_y:lip2_y2,lip2_x:lip2_x2].shape[0])))
    else:
        lip_img = lip_img[:,:,::-1].copy()
        lip_img = align_faces(lip_img)
        lip_img = lip_img[0]
        lip_x,lip_y,lip_x2,lip_y2,eye_x_l,eye_y_l,eye_x2_l,eye_y2_l,eye_x_r,eye_y_r,eye_x2_r,eye_y2_r = tracker(lip_img)
        lip2_x,lip2_y,lip2_x2,lip2_y2,eye2_x_l,eye2_y_l,eye2_x2_l,eye2_y2_l,eye2_x_r,eye2_y_r,eye2_x2_r,eye2_y2_r = tracker(img)
        img[lip2_y:lip2_y2,lip2_x:lip2_x2] = cv2.resize(lip_img[lip_y:lip_y2,lip_x:lip_x2],(int(img[lip2_y:lip2_y2,lip2_x:lip2_x2].shape[1]),int(img[lip2_y:lip2_y2,lip2_x:lip2_x2].shape[0])))
    
    if eye_img is None:
        if skin_img is None:
            img = img.copy()
        else:
            lip_x,lip_y,lip_x2,lip_y2,eye_x_l,eye_y_l,eye_x2_l,eye_y2_l,eye_x_r,eye_y_r,eye_x2_r,eye_y2_r = tracker(src_img)
            lip2_x,lip2_y,lip2_x2,lip2_y2,eye2_x_l,eye2_y_l,eye2_x2_l,eye2_y2_l,eye2_x_r,eye2_y_r,eye2_x2_r,eye2_y2_r = tracker(img)
            img[eye2_y_l:eye2_y2_l,eye2_x_l:eye2_x2_l] = cv2.resize(src_img[eye_y_l:eye_y2_l,eye_x_l:eye_x2_l],(int(img[eye2_y_l:eye2_y2_l,eye2_x_l:eye2_x2_l].shape[1]),int(img[eye2_y_l:eye2_y2_l,eye2_x_l:eye2_x2_l].shape[0])))
            img[eye2_y_r:eye2_y2_r,eye2_x_r:eye2_x2_r] = cv2.resize(src_img[eye_y_r:eye_y2_r,eye_x_r:eye_x2_r],(int(img[eye2_y_r:eye2_y2_r,eye2_x_r:eye2_x2_r].shape[1]),int(img[eye2_y_r:eye2_y2_r,eye2_x_r:eye2_x2_r].shape[0])))
    else:
        eye_img = eye_img[:,:,::-1].copy()
        eye_img = align_faces(eye_img)
        eye_img = eye_img[0]
        lip_x,lip_y,lip_x2,lip_y2,eye_x_l,eye_y_l,eye_x2_l,eye_y2_l,eye_x_r,eye_y_r,eye_x2_r,eye_y2_r = tracker(eye_img)
        lip2_x,lip2_y,lip2_x2,lip2_y2,eye2_x_l,eye2_y_l,eye2_x2_l,eye2_y2_l,eye2_x_r,eye2_y_r,eye2_x2_r,eye2_y2_r = tracker(img)
        img[eye2_y_l:eye2_y2_l,eye2_x_l:eye2_x2_l] = cv2.resize(eye_img[eye_y_l:eye_y2_l,eye_x_l:eye_x2_l],(int(img[eye2_y_l:eye2_y2_l,eye2_x_l:eye2_x2_l].shape[1]),int(img[eye2_y_l:eye2_y2_l,eye2_x_l:eye2_x2_l].shape[0])))
        img[eye2_y_r:eye2_y2_r,eye2_x_r:eye2_x2_r] = cv2.resize(eye_img[eye_y_r:eye_y2_r,eye_x_r:eye_x2_r],(int(img[eye2_y_r:eye2_y2_r,eye2_x_r:eye2_x2_r].shape[1]),int(img[eye2_y_r:eye2_y2_r,eye2_x_r:eye2_x2_r].shape[0])))
    return img


def full_makeup(frame ,skin_image=None, lip_image=None, eye_image=None):    
    #cap.set(3,720)
    #cap.set(4,1080)
    frame = cv2.resize(frame,(int(256),int(256)))
    temp = frame[:,:,::-1].copy()  #
    dets = detector(temp, 1)
    if len(dets) != 0:
        temp_face = align_faces(temp)
        src_img = temp_face[0]

        X_img = preprocess(src_img)
        X_img = np.expand_dims(X_img, axis=0)

        ###sess run이 오래 걸림....0.55초 정도->gpu(gtx 1080)은 약 11배 

        #src_img, output_img로 
        faces1 = detector(src_img,1)
        if len(faces1) !=0:

            ######아예 새로운img만들기########

            img = make_img(src_img,skin_img = skin_image,lip_img =lip_image ,eye_img= eye_image)

            #img = img[:,:,::-1].copy()
            Y_img = preprocess(img)
            Y_img = np.expand_dims(Y_img, axis=0)
 
            output = sess.run(Xs, feed_dict={
                X: X_img,
                Y: Y_img
            })
            output_img = postprocess(output[0])
            final_output = output_img[:,:,::-1].copy()
            
            #그냥 화면
            Y_img = preprocess(src_img)
            Y_img = np.expand_dims(Y_img, axis=0)
            output = sess.run(Xs, feed_dict={
                X: X_img,
                Y: Y_img
            })
            output_img = postprocess(output[0])
            final_output_no = output_img[:,:,::-1].copy()
           
        key = cv2.waitKey(1)
    else:
        frame = cv2.resize(frame, (int(256), int(256)))
    return frame, final_output_no, final_output




