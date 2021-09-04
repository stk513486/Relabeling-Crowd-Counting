# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math



def Mixture_Bayesian_Loss(z = None , GT = None , score_scale = None , BG = None  , Version = None , delta = None , sigma = None , training_stage = "1"):

    pi = tf.constant(math.pi)


    cx = tf.range(start = 0 , limit = tf.shape(z)[2] )
    cy = tf.range(start = 0 , limit = tf.shape(z)[1] ) 

    cx = tf.cast(cx,tf.float32)
    cy = tf.cast(cy,tf.float32)
    

    coord_x , coord_y = tf.meshgrid(cx , cy)   

    cx = cx[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:] 
    cy = cy[tf.newaxis,tf.newaxis,tf.newaxis,:,tf.newaxis] 
    
    
    gt_x = GT[:,:,:,0:1] 
    gt_y = GT[:,:,:,1:2]
    gt_w = GT[:,:,:,2:3]
    
    gt_x = gt_x[:,:,:,:,tf.newaxis]       
    gt_y = gt_y[:,:,:,:,tf.newaxis]
    gt_w = gt_w[:,:,:,:,tf.newaxis]


    dx = cx - gt_x      
    dy = cy - gt_y 

    dx = tf.multiply(dx , dx)
    dy = tf.multiply(dy , dy)

    
    dis = dx + dy
    if training_stage == "1":
        pass
    else:
        sigma = sigma[:,:,:,tf.newaxis , tf.newaxis]       

    dis = - dis / (2 * tf.multiply(sigma , sigma))  

    
    dis = tf.exp(dis)
    dis = dis * 1 / (2*pi*tf.multiply(sigma,sigma))
     
    gt_w = -1 * tf.multiply(score_scale ,  gt_w)
    gt_w = tf.nn.softmax(gt_w ,axis = 2)
        
    dis = tf.multiply(dis , gt_w)
    dis = tf.reduce_sum(dis , axis = [2] , keepdims = False)

    

    
    if BG == "uni":
        background = tf.ones(shape = [1,1,tf.shape(z)[1] , tf.shape(z)[2]]) / (tf.cast(tf.shape(z)[1] * tf.shape(z)[2],tf.float32))

    elif BG == "adap":
    
        dis_sum = tf.reduce_sum(dis , axis=[1] , keepdims = True)
        dis_sum = tf.clip_by_value(dis_sum , 0.0 , 1.0)
        background = 1 - dis_sum
        background = background / tf.reduce_sum(background)

    
    dis = tf.concat([dis , background] , axis = 1)
    prob = dis / (tf.reduce_sum(dis , axis=[1] , keepdims = True))

    z = tf.transpose(z , perm=[0,3,1,2])  
    decoupled_density = tf.multiply(z , prob)


    background_density = decoupled_density[:,-1:,:,:]
    decoupled_density = decoupled_density[:,:-1,:,:]    
    
    loss = tf.reduce_sum(tf.abs(tf.reduce_sum(decoupled_density , axis=[2,3] ,keepdims=False) - 1) , axis=[1],keepdims=False)
    loss = tf.reduce_mean(loss , axis = [0])

    zero_count = tf.reduce_mean(tf.abs(tf.reduce_sum(background_density , axis=[2,3] ,keepdims=False) - 0) , axis=[0,1],keepdims=False)
    loss = loss + zero_count
    
    

    grid = tf.concat([coord_x[:,:,tf.newaxis] , coord_y[:,:,tf.newaxis]] , axis = 2)[tf.newaxis,tf.newaxis,:,:,:]

    decoupled_density_normal = decoupled_density / tf.reduce_sum(decoupled_density , axis = [2,3] , keepdims = True)      
    expectation = tf.multiply(decoupled_density_normal[:,:,:,:,tf.newaxis]  , grid)                         
    
    expectation = tf.reduce_sum(expectation , axis = [2,3] , keepdims = False)

    
    
    gt_loc = tf.concat([gt_x , gt_y] , axis = 3)
    gt_loc = tf.multiply(gt_w , gt_loc)   
    gt_loc = tf.reduce_sum(gt_loc , axis = [2,4],keepdims = False)
    

    loc_loss = tf.reduce_sum(tf.nn.relu(tf.abs(expectation - gt_loc) - delta[1]) , axis = [1,2] , keepdims = False)
    loc_loss = tf.reduce_mean(loc_loss , axis = 0)   


    if Version == "V1":

        pass

    elif Version == "V2":
        
        loss = loss + delta[0] * loc_loss
    
    return loss , tf.reduce_sum(decoupled_density , axis = [2,3] , keepdims = False) , expectation



def compute_MAE_and_MSE(z , Count):
    
    predict = tf.reduce_sum(z , axis = [1,2,3])    
    
    difference = ( predict - Count)
    
    MAE = tf.reduce_sum(tf.abs(difference))
    MSE = tf.reduce_sum(tf.square(difference))
    
    return (MAE , MSE)






###  Re-estimate,  only  for  Inference_for_re-estimate.py
def Re_estimate(z , GT , score_scale):

    sigma = 2
    pi = tf.constant(math.pi)
    
    
  
    cx = tf.range(start = 0 , limit = tf.shape(z)[2] )
    cy = tf.range(start = 0 , limit = tf.shape(z)[1] ) 

    cx = tf.cast(cx,tf.float32)
    cy = tf.cast(cy,tf.float32)
    

    coord_x , coord_y = tf.meshgrid(cx , cy)   


    cx = cx[tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis,:] 
    cy = cy[tf.newaxis,tf.newaxis,tf.newaxis,:,tf.newaxis] 
    
    
    gt_x = GT[:,:,:,0:1]
    gt_y = GT[:,:,:,1:2]
    gt_w = GT[:,:,:,2:3]
    
    gt_x = gt_x[:,:,:,:,tf.newaxis]    
    gt_y = gt_y[:,:,:,:,tf.newaxis]
    gt_w = gt_w[:,:,:,:,tf.newaxis]


    dx = cx - gt_x      
    dy = cy - gt_y    

    dx = tf.multiply(dx , dx)
    dy = tf.multiply(dy , dy)

    
    dis = dx + dy                
    dis = - dis / (2 * sigma**2) 
    
    dis = tf.exp(dis)
    dis = dis * 1 / (2*pi * sigma**2)
    
    
    gt_w = -1 * tf.multiply(score_scale ,  gt_w)
    gt_w = tf.nn.softmax(gt_w ,axis = 2) 
        
    dis = tf.multiply(dis , gt_w)
    dis = tf.reduce_sum(dis , axis = [2] , keepdims = False)



    dis_sum = tf.reduce_sum(dis , axis=[1] , keepdims = True)
    dis_sum = tf.clip_by_value(dis_sum , 0.0 , 1.0)
    background = 1 - dis_sum
    background = background / tf.reduce_sum(background)
    
    dis = tf.concat([dis , background] , axis = 1)
    prob = dis / (tf.reduce_sum(dis , axis=[1] , keepdims = True))

    z = tf.transpose(z , perm=[0,3,1,2])  
    decoupled_density = tf.multiply(z , prob)

    background_density = decoupled_density[:,-1:,:,:]
    decoupled_density = decoupled_density[:,:-1,:,:]    
    
    loss = tf.reduce_sum(tf.abs(tf.reduce_sum(decoupled_density , axis=[2,3] ,keepdims=False) - 1) , axis=[1],keepdims=False)
    loss = tf.reduce_mean(loss , axis = [0])

    zero_count = tf.reduce_mean(tf.abs(tf.reduce_sum(background_density , axis=[2,3] ,keepdims=False) - 0) , axis=[0,1],keepdims=False)
    loss = loss + zero_count
    
    

    grid = tf.concat([coord_x[:,:,tf.newaxis] , coord_y[:,:,tf.newaxis]] , axis = 2)[tf.newaxis,tf.newaxis,:,:,:]

    decoupled_density_normal = decoupled_density / tf.reduce_sum(decoupled_density , axis = [2,3] , keepdims = True)
    expectation = tf.multiply(decoupled_density_normal[:,:,:,:,tf.newaxis]  , grid)

    expectation = tf.reduce_sum(expectation , axis = [2,3] , keepdims = False)

    
    grid_minus_mean = grid - expectation[:,:,tf.newaxis , tf.newaxis , :]

    
    x_u_2 = tf.multiply(grid_minus_mean , grid_minus_mean)

    
    variance = tf.multiply(decoupled_density_normal[:,:,:,:,tf.newaxis] ,  x_u_2)  
    variance = tf.reduce_sum(variance , axis = [2,3] , keepdims = False)
  
    
    gt_loc = tf.concat([gt_x , gt_y] , axis = 3)
    gt_loc = tf.multiply(gt_w , gt_loc)
    gt_loc = tf.reduce_sum(gt_loc , axis = [2,4],keepdims = False)
   


    loc_loss = tf.reduce_sum(tf.nn.relu(tf.abs(expectation - gt_loc) - 0.5) , axis = [1,2] , keepdims = False)
    loc_loss = tf.reduce_mean(loc_loss , axis = 0)   
    

    loss = loss + 0.25 * loc_loss
    
    return loss , tf.reduce_sum(decoupled_density , axis = [2,3] , keepdims = False) , expectation , variance



