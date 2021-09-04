# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:35:25 2019

@author: a9090
"""


import tensorflow as tf
import numpy as np
import cv2
import math


def compute_MAE_and_MSE(z , Count):
    
    predict = tf.reduce_sum(z , axis = [1,2,3])    
    
    difference = ( predict - Count)
    
    MAE = tf.reduce_sum(tf.abs(difference))
    MSE = tf.reduce_sum(tf.square(difference))
    
    return (MAE , MSE)




def Bayesian_Loss_V2(z = None , GT = None):

    pi = tf.constant(math.pi)

  
    cx = tf.range(start = 0 , limit = tf.shape(z)[2] )
    cy = tf.range(start = 0 , limit = tf.shape(z)[1] ) 

    cx = tf.cast(cx,tf.float32)
    cy = tf.cast(cy,tf.float32)
    
    ###
    coord_x , coord_y = tf.meshgrid(cx , cy)   
    ###

    cx = cx[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:] # 1,1,1,1,128
    cy = cy[tf.newaxis,tf.newaxis,tf.newaxis,:,tf.newaxis] # 1,1,1, 96,1
    
    
    gt_x = GT[:,:,:,0:1]   # 1,100,10,1
    gt_y = GT[:,:,:,1:2]


    gt_x = gt_x[:,:,:,:,tf.newaxis]         #  1,100,10,1,1
    gt_y = gt_y[:,:,:,:,tf.newaxis]


    dx = cx - gt_x      #  1,100,10,1,128
    dy = cy - gt_y      #  1,100,10,96,1

    dx = tf.multiply(dx , dx)
    dy = tf.multiply(dy , dy)
    sigma = get_sigma()

    
    dis = dx + dy
    tmp_dis = dis
    dis = - dis / (2 * sigma**2)   # 1,100,10,96,128
    
    dis = tf.exp(dis)
    dis = dis * 1 / (2*pi*sigma**2)
    

    dis = tf.reduce_sum(dis , axis = [2] , keepdims = False)
    # 1 100  96 128


    ###
    # for background
    tmp_dis = tf.reduce_sum(tmp_dis , axis = [2] , keepdims = False)
    
    bg_D = 0.1 * 0.5 * (tf.cast(tf.shape(z)[1] + tf.shape(z)[2] , tf.float32))

    bg_min = tf.reduce_min(tmp_dis , axis = [1] , keepdims = True)
    bg_min = tf.clip_by_value(bg_min , 0.0 , 10000.0)

    bg_dis = bg_D - tf.sqrt(bg_min)
    bg_dis = tf.multiply(bg_dis , bg_dis)
    bg_dis = - bg_dis / (2*sigma**2)
    bg_dis = tf.exp(bg_dis)
    bg_dis = bg_dis * 1 / (2*pi*sigma**2)

    ###

    dis = tf.concat([dis , bg_dis] , axis = 1)

    prob = dis / (tf.reduce_sum(dis , axis=[1] , keepdims = True) + 1e-16)

    z = tf.transpose(z , perm=[0,3,1,2])  
    decoupled_density = tf.multiply(z , prob)
    # 1  101  96  128

    background_density = decoupled_density[:,-1:,:,:]
    decoupled_density = decoupled_density[:,:-1,:,:]    



    loss = tf.reduce_sum(tf.abs(tf.reduce_sum(decoupled_density , axis=[2,3] ,keepdims=False) - 1) , axis=[1],keepdims=False)
    loss = tf.reduce_mean(loss , axis = [0])

    zero_count = tf.reduce_mean(tf.abs(tf.reduce_sum(background_density , axis=[2,3] ,keepdims=False) - 0) , axis=[0,1],keepdims=False)
    loss = loss + zero_count
    
    
    return loss






def pixel_wise_l2_loss(z = None , GT = None):

    pi = tf.constant(math.pi)
    
  
    cx = tf.range(start = 0 , limit = tf.shape(z)[2] )
    cy = tf.range(start = 0 , limit = tf.shape(z)[1] ) 

    cx = tf.cast(cx,tf.float32)
    cy = tf.cast(cy,tf.float32)
    
    ###
    coord_x , coord_y = tf.meshgrid(cx , cy)   
    ###

    cx = cx[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:] # 1,1,1,1,128
    cy = cy[tf.newaxis,tf.newaxis,tf.newaxis,:,tf.newaxis] # 1,1,1, 96,1
    
    
    gt_x = GT[:,:,:,0:1]   # 1,100,10,1
    gt_y = GT[:,:,:,1:2]

    
    gt_x = gt_x[:,:,:,:,tf.newaxis]         #  1,100,10,1,1
    gt_y = gt_y[:,:,:,:,tf.newaxis]



    dx = cx - gt_x      #  1,100,10,1,128
    dy = cy - gt_y      #  1,100,10,96,1

    dx = tf.multiply(dx , dx)
    dy = tf.multiply(dy , dy)
    sigma = get_sigma()

    
    dis = dx + dy
    dis = - dis / (2 * sigma**2)   # 1,100,10,96,128
    
    dis = tf.exp(dis)
    dis = dis * 1 / (2*pi*sigma**2)
    

    dis = tf.reduce_sum(dis , axis = [2] , keepdims = False)
    # 1 100  96 128


    dis = tf.reduce_sum(dis , axis=[1] , keepdims = True)
    dis = dis / tf.reduce_sum(dis , axis = [1,2,3]) * tf.cast(tf.shape(GT)[1] , tf.float32)

    z = tf.transpose(z , perm=[0,3,1,2])  
    # 1  101  96  128

    a = z - dis


    loss = 0.5*tf.reduce_mean(tf.reduce_sum( tf.multiply(a , a), axis = [1,2,3] ,keepdims=False) , axis = [0] , keepdims = False)

    
    return loss



def get_sigma():
    return 8


def ADSC_Loss(z = None , GT = None):

    pi = tf.constant(math.pi)
    
  
    cx = tf.range(start = 0 , limit = tf.shape(z)[2] )
    cy = tf.range(start = 0 , limit = tf.shape(z)[1] ) 

    cx = tf.cast(cx,tf.float32)
    cy = tf.cast(cy,tf.float32)
    
    ###
    coord_x , coord_y = tf.meshgrid(cx , cy)   
    ###

    cx = cx[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:] # 1,1,1,1,128
    cy = cy[tf.newaxis,tf.newaxis,tf.newaxis,:,tf.newaxis] # 1,1,1, 96,1
    
    
    gt_x = GT[:,:,:,0:1]   # 1,100,10,1
    gt_y = GT[:,:,:,1:2]
    
    gt_x = gt_x[:,:,:,:,tf.newaxis]         #  1,100,10,1,1
    gt_y = gt_y[:,:,:,:,tf.newaxis]


    dx = cx - gt_x      #  1,100,10,1,128
    dy = cy - gt_y      #  1,100,10,96,1

    dx = tf.multiply(dx , dx)
    dy = tf.multiply(dy , dy)
    sigma = get_sigma()
    
    dis = dx + dy                
    dis = - dis / (2 * sigma**2)   # 1,100,10,96,128
    
    dis = tf.exp(dis)
    dis = dis * 1 / (2*pi*sigma**2)
    
    
    dis = tf.reduce_sum(dis , axis = [2] , keepdims = False)
    # 1 100  96 128
    

    prob = dis / (tf.reduce_sum(dis , axis=[1] , keepdims = True) + 1e-16)


    z = tf.transpose(z , perm=[0,3,1,2])

    dis = tf.reduce_sum(dis , axis=[1] , keepdims = True)
    dis = dis / tf.reduce_sum(dis , axis = [1,2,3]) * tf.cast(tf.shape(GT)[1] , tf.float32)

    a = z - dis

    loss = 0.5*tf.reduce_mean(tf.reduce_sum( tf.multiply(a , a), axis = [1,2,3] ,keepdims=False) , axis = [0] , keepdims = False)

    decoupled_density = tf.multiply(z , prob)

    ###
    grid = tf.concat([coord_x[:,:,tf.newaxis] , coord_y[:,:,tf.newaxis]] , axis = 2)[tf.newaxis,tf.newaxis,:,:,:]

    decoupled_density_normal = decoupled_density / tf.reduce_sum(decoupled_density , axis = [2,3] , keepdims = True)      #  1 100 96 128 1
    expectation = tf.multiply(decoupled_density_normal[:,:,:,:,tf.newaxis]  , grid)                                       #  1 100 96 128 2
    
    expectation = tf.reduce_sum(expectation , axis = [2,3] , keepdims = False)
    ###
    
    return loss , tf.reduce_sum(decoupled_density , axis = [2,3] , keepdims = False) , expectation
                    # 1 100                                                              1 100 2
