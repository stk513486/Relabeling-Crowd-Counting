# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np




def common_conv2d(z , in_filter = None , out_filter = None ,Name = None):

    with tf.variable_scope(Name):    
        W = tf.compat.v1.get_variable(name = Name+"_W" , shape = [1,1,in_filter,out_filter])
        b = tf.compat.v1.get_variable(name = Name+"_b" , shape = [out_filter],initializer = tf.compat.v1.zeros_initializer())
    
        z = tf.compat.v1.nn.conv2d(z , W , strides=[1,1,1,1] ,  padding="SAME") + b
        z = tf.nn.relu(z)    
        
        return z
    
    
def dilated_conv2d(z , in_filter = None , out_filter = None , dilated_rate = None,Name = None):

    with tf.variable_scope(Name):    
        W = tf.compat.v1.get_variable(name = Name+"_W" , shape = [3,3,in_filter,out_filter])
        b = tf.compat.v1.get_variable(name = Name+"_b" , shape = [out_filter],initializer = tf.compat.v1.zeros_initializer())
    
        z = tf.nn.atrous_conv2d(z , W , rate = dilated_rate , padding="SAME") + b
        z = tf.nn.relu(z)    
        
        return z


def CAN_module(z , scale = None):

    z_pool = tf.nn.avg_pool2d(z , ksize = [1,scale,scale,1] , strides = [1,scale,scale,1] , padding='SAME')                    

    z_pool = tf.layers.conv2d(z_pool , 1 , [1,1] , strides=[1,1] , padding="SAME")

    z_pool = tf.compat.v1.image.resize_bilinear(z_pool , [tf.shape(z)[1] , tf.shape(z)[2]]) 

    return z_pool
        



class VGG19():  # VGG-19 model
    
    def __init__(self,vgg_file = "./vgg19.npy" ):

        self.param_dict = np.load(vgg_file , allow_pickle=True ,  encoding='latin1').item()
        
        "-------------------------------------------------------"

        self.W_1_1 = tf.Variable(self.param_dict["conv1_1"][0])
        self.b_1_1 = tf.Variable(self.param_dict["conv1_1"][1])
        
        self.W_1_2 = tf.Variable(self.param_dict["conv1_2"][0])

        self.b_1_2 = tf.Variable(self.param_dict["conv1_2"][1])
        "-------------------------------------------------------"        

        self.W_2_1 = tf.Variable(self.param_dict["conv2_1"][0])
        self.b_2_1 = tf.Variable(self.param_dict["conv2_1"][1])
        
        self.W_2_2 = tf.Variable(self.param_dict["conv2_2"][0])

        self.b_2_2 = tf.Variable(self.param_dict["conv2_2"][1])
        "-------------------------------------------------------"        

        self.W_3_1 = tf.Variable(self.param_dict["conv3_1"][0])
        self.b_3_1 = tf.Variable(self.param_dict["conv3_1"][1])

        self.W_3_2 = tf.Variable(self.param_dict["conv3_2"][0])
        self.b_3_2 = tf.Variable(self.param_dict["conv3_2"][1])

        self.W_3_3 = tf.Variable(self.param_dict["conv3_3"][0])
        self.b_3_3 = tf.Variable(self.param_dict["conv3_3"][1])

        self.W_3_4 = tf.Variable(self.param_dict["conv3_4"][0])

        self.b_3_4 = tf.Variable(self.param_dict["conv3_4"][1])
        "-------------------------------------------------------"              

        self.W_4_1 = tf.Variable(self.param_dict["conv4_1"][0])
        self.b_4_1 = tf.Variable(self.param_dict["conv4_1"][1])
        
        self.W_4_2 = tf.Variable(self.param_dict["conv4_2"][0])
        self.b_4_2 = tf.Variable(self.param_dict["conv4_2"][1])
        
        self.W_4_3 = tf.Variable(self.param_dict["conv4_3"][0])
        self.b_4_3 = tf.Variable(self.param_dict["conv4_3"][1])
        
        self.W_4_4 = tf.Variable(self.param_dict["conv4_4"][0])
        self.b_4_4 = tf.Variable(self.param_dict["conv4_4"][1])

        "-------------------------------------------------------"

        self.W_5_1 = tf.Variable(self.param_dict["conv5_1"][0])
        self.b_5_1 = tf.Variable(self.param_dict["conv5_1"][1])
        
        self.W_5_2 = tf.Variable(self.param_dict["conv5_2"][0])
        self.b_5_2 = tf.Variable(self.param_dict["conv5_2"][1])
        
        self.W_5_3 = tf.Variable(self.param_dict["conv5_3"][0])
        self.b_5_3 = tf.Variable(self.param_dict["conv5_3"][1])
        
        self.W_5_4 = tf.Variable(self.param_dict["conv5_4"][0])
        self.b_5_4 = tf.Variable(self.param_dict["conv5_4"][1])
        
        "-------------------------------------------------------"
       
        
    def forward(self , x ):
        
        with tf.variable_scope("VGG_backbone"):
            z = tf.nn.conv2d(x , self.W_1_1 , [1,1,1,1] , padding = "SAME") + self.b_1_1
            z = tf.nn.relu(z)    
            z = tf.nn.conv2d(z , self.W_1_2 , [1,1,1,1] , padding = "SAME") + self.b_1_2
            z = tf.nn.relu(z)
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




            z = tf.nn.conv2d(z , self.W_2_1 , [1,1,1,1] , padding = "SAME") + self.b_2_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_2_2 , [1,1,1,1] , padding = "SAME") + self.b_2_2
            z = tf.nn.relu(z)
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



            z = tf.nn.conv2d(z , self.W_3_1 , [1,1,1,1] , padding = "SAME") + self.b_3_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_3_2 , [1,1,1,1] , padding = "SAME") + self.b_3_2
            z = tf.nn.relu(z)        
            z = tf.nn.conv2d(z , self.W_3_3 , [1,1,1,1] , padding = "SAME") + self.b_3_3
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_3_4 , [1,1,1,1] , padding = "SAME") + self.b_3_4
            z = tf.nn.relu(z)                        
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        
        
            z = tf.nn.conv2d(z , self.W_4_1 , [1,1,1,1] , padding = "SAME") + self.b_4_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_4_2 , [1,1,1,1] , padding = "SAME") + self.b_4_2
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_4_3 , [1,1,1,1] , padding = "SAME") + self.b_4_3
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_4_4 , [1,1,1,1] , padding = "SAME") + self.b_4_4
            z = tf.nn.relu(z)
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    

            z = tf.nn.conv2d(z , self.W_5_1 , [1,1,1,1] , padding = "SAME") + self.b_5_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_5_2 , [1,1,1,1] , padding = "SAME") + self.b_5_2
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_5_3 , [1,1,1,1] , padding = "SAME") + self.b_5_3
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_5_4 , [1,1,1,1] , padding = "SAME") + self.b_5_4
            z = tf.nn.relu(z)


            z = tf.compat.v1.image.resize_bilinear(z , [ 2 * tf.shape(z)[1] , 2 * tf.shape(z)[2]])        
        
            z = dilated_conv2d(z , 512 , 256 , 1 , "128")
            z = dilated_conv2d(z , 256 , 128 , 1 , "64")
            z = tf.layers.conv2d(z , 1 , [1,1] , strides=[1,1] , padding="SAME")


            return z


#-----------------------------------------------------------------------------------------------


class CSRNet(): # for CSRNet
    
    def __init__(self,vgg_file = "./vgg16.npy" ):

        self.param_dict = np.load(vgg_file , allow_pickle=True ,  encoding='latin1').item()
        
        self.W_1_1 = tf.Variable(self.param_dict["conv1_1"][0])
        self.b_1_1 = tf.Variable(self.param_dict["conv1_1"][1])
        
        self.W_1_2 = tf.Variable(self.param_dict["conv1_2"][0])
        self.b_1_2 = tf.Variable(self.param_dict["conv1_2"][1])
        
        self.W_2_1 = tf.Variable(self.param_dict["conv2_1"][0])
        self.b_2_1 = tf.Variable(self.param_dict["conv2_1"][1])
        
        self.W_2_2 = tf.Variable(self.param_dict["conv2_2"][0])
        self.b_2_2 = tf.Variable(self.param_dict["conv2_2"][1])
        
        self.W_3_1 = tf.Variable(self.param_dict["conv3_1"][0])
        self.b_3_1 = tf.Variable(self.param_dict["conv3_1"][1])

        self.W_3_2 = tf.Variable(self.param_dict["conv3_2"][0])
        self.b_3_2 = tf.Variable(self.param_dict["conv3_2"][1])

        self.W_3_3 = tf.Variable(self.param_dict["conv3_3"][0])
        self.b_3_3 = tf.Variable(self.param_dict["conv3_3"][1])
        
        self.W_4_1 = tf.Variable(self.param_dict["conv4_1"][0])
        self.b_4_1 = tf.Variable(self.param_dict["conv4_1"][1])
        
        self.W_4_2 = tf.Variable(self.param_dict["conv4_2"][0])
        self.b_4_2 = tf.Variable(self.param_dict["conv4_2"][1])
        
        self.W_4_3 = tf.Variable(self.param_dict["conv4_3"][0])
        self.b_4_3 = tf.Variable(self.param_dict["conv4_3"][1])
        
        
        
    def forward(self , x ):
        
        with tf.variable_scope("VGG_backbone"):
            z = tf.nn.conv2d(x , self.W_1_1 , [1,1,1,1] , padding = "SAME") + self.b_1_1
            z = tf.nn.relu(z)    
            z = tf.nn.conv2d(z , self.W_1_2 , [1,1,1,1] , padding = "SAME") + self.b_1_2
            z = tf.nn.relu(z)
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




            z = tf.nn.conv2d(z , self.W_2_1 , [1,1,1,1] , padding = "SAME") + self.b_2_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_2_2 , [1,1,1,1] , padding = "SAME") + self.b_2_2
            z = tf.nn.relu(z)
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



            z = tf.nn.conv2d(z , self.W_3_1 , [1,1,1,1] , padding = "SAME") + self.b_3_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_3_2 , [1,1,1,1] , padding = "SAME") + self.b_3_2
            z = tf.nn.relu(z)        
            z = tf.nn.conv2d(z , self.W_3_3 , [1,1,1,1] , padding = "SAME") + self.b_3_3
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        
        
            z = tf.nn.conv2d(z , self.W_4_1 , [1,1,1,1] , padding = "SAME") + self.b_4_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_4_2 , [1,1,1,1] , padding = "SAME") + self.b_4_2
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_4_3 , [1,1,1,1] , padding = "SAME") + self.b_4_3
            z = tf.nn.relu(z)

            z = dilated_conv2d(z , 512 , 512 , 2 , "back_end_1")
            z = dilated_conv2d(z , 512 , 512 , 2 , "back_end_2")
            z = dilated_conv2d(z , 512 , 512 , 2 , "back_end_3")
            z = dilated_conv2d(z , 512 , 256 , 2 , "back_end_4")
            z = dilated_conv2d(z , 256 , 128 , 2 , "back_end_5")
            z = dilated_conv2d(z , 128 ,  64 , 2 , "back_end_6")

            z = tf.layers.conv2d(z , 1 , [1,1] , strides=[1,1] , padding="SAME")
        
            return z



#-----------------------------------------------------------------------------------------------


class CAN(): # for CAN
    
    def __init__(self,vgg_file = "./vgg16.npy" ):

        self.param_dict = np.load(vgg_file , allow_pickle=True ,  encoding='latin1').item()
        
        self.W_1_1 = tf.Variable(self.param_dict["conv1_1"][0])
        self.b_1_1 = tf.Variable(self.param_dict["conv1_1"][1])
        
        self.W_1_2 = tf.Variable(self.param_dict["conv1_2"][0])
        self.b_1_2 = tf.Variable(self.param_dict["conv1_2"][1])
        
        self.W_2_1 = tf.Variable(self.param_dict["conv2_1"][0])
        self.b_2_1 = tf.Variable(self.param_dict["conv2_1"][1])
        
        self.W_2_2 = tf.Variable(self.param_dict["conv2_2"][0])
        self.b_2_2 = tf.Variable(self.param_dict["conv2_2"][1])
        
        self.W_3_1 = tf.Variable(self.param_dict["conv3_1"][0])
        self.b_3_1 = tf.Variable(self.param_dict["conv3_1"][1])

        self.W_3_2 = tf.Variable(self.param_dict["conv3_2"][0])
        self.b_3_2 = tf.Variable(self.param_dict["conv3_2"][1])

        self.W_3_3 = tf.Variable(self.param_dict["conv3_3"][0])
        self.b_3_3 = tf.Variable(self.param_dict["conv3_3"][1])
        
        self.W_4_1 = tf.Variable(self.param_dict["conv4_1"][0])
        self.b_4_1 = tf.Variable(self.param_dict["conv4_1"][1])
        
        self.W_4_2 = tf.Variable(self.param_dict["conv4_2"][0])
        self.b_4_2 = tf.Variable(self.param_dict["conv4_2"][1])
        
        self.W_4_3 = tf.Variable(self.param_dict["conv4_3"][0])
        self.b_4_3 = tf.Variable(self.param_dict["conv4_3"][1])
        
        
        
    def forward(self , x ):
        
        with tf.variable_scope("VGG_backbone"):
            z = tf.nn.conv2d(x , self.W_1_1 , [1,1,1,1] , padding = "SAME") + self.b_1_1
            z = tf.nn.relu(z)    
            z = tf.nn.conv2d(z , self.W_1_2 , [1,1,1,1] , padding = "SAME") + self.b_1_2
            z = tf.nn.relu(z)
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




            z = tf.nn.conv2d(z , self.W_2_1 , [1,1,1,1] , padding = "SAME") + self.b_2_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_2_2 , [1,1,1,1] , padding = "SAME") + self.b_2_2
            z = tf.nn.relu(z)
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



            z = tf.nn.conv2d(z , self.W_3_1 , [1,1,1,1] , padding = "SAME") + self.b_3_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_3_2 , [1,1,1,1] , padding = "SAME") + self.b_3_2
            z = tf.nn.relu(z)        
            z = tf.nn.conv2d(z , self.W_3_3 , [1,1,1,1] , padding = "SAME") + self.b_3_3
            z = tf.nn.max_pool2d(z, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        
        
            z = tf.nn.conv2d(z , self.W_4_1 , [1,1,1,1] , padding = "SAME") + self.b_4_1
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_4_2 , [1,1,1,1] , padding = "SAME") + self.b_4_2
            z = tf.nn.relu(z)
            z = tf.nn.conv2d(z , self.W_4_3 , [1,1,1,1] , padding = "SAME") + self.b_4_3
            z = tf.nn.relu(z)

            s1 = CAN_module(z , 1)
            s2 = CAN_module(z , 2)
            s3 = CAN_module(z , 3)
            s4 = CAN_module(z , 6)

            c1 = s1 - z
            c2 = s2 - z
            c3 = s3 - z
            c4 = s4 - z

            w1 = tf.nn.sigmoid(tf.layers.conv2d(c1 , 512 , [1,1] , strides=[1,1] , padding="SAME"))
            w2 = tf.nn.sigmoid(tf.layers.conv2d(c2 , 512 , [1,1] , strides=[1,1] , padding="SAME"))
            w3 = tf.nn.sigmoid(tf.layers.conv2d(c3 , 512 , [1,1] , strides=[1,1] , padding="SAME"))
            w4 = tf.nn.sigmoid(tf.layers.conv2d(c4 , 512 , [1,1] , strides=[1,1] , padding="SAME"))

            sc1 = tf.multiply(w1,s1)
            sc2 = tf.multiply(w2,s2)
            sc3 = tf.multiply(w3,s3)
            sc4 = tf.multiply(w4,s4)

            scale_context = (sc1 + sc2 + sc3 + sc4) / (w1 + w2 + w3 + w4 + 1e-10)

            zz = tf.concat([z , scale_context] , axis = 3)
          
            z = zz
            
            z = dilated_conv2d(z ,1024 , 512 , 2 , "back_end_1")
            z = dilated_conv2d(z , 512 , 512 , 2 , "back_end_2")
            z = dilated_conv2d(z , 512 , 512 , 2 , "back_end_3")
            z = dilated_conv2d(z , 512 , 256 , 2 , "back_end_4")
            z = dilated_conv2d(z , 256 , 128 , 2 , "back_end_5")
            z = dilated_conv2d(z , 128 ,  64 , 2 , "back_end_6")

            z = tf.layers.conv2d(z , 1 , [1,1] , strides=[1,1] , padding="SAME")

        
            return z


