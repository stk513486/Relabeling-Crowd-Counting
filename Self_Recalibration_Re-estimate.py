# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.io as scio
import scipy
from PIL import Image

import time
import random
import os

from models import VGG19 , CSRNet , CAN

import argparse
import utils.tools as tools
import utils.relabel_tools as relabel_tools
import data_loader as Shanghai




    


        

'''
--------------------------------------------------------------------------------------------------------------------------        

'''

def Training(fold = None , percentage = None , dataset = None , model = None):
    
    Epoch = 0

    scale = np.array([5])

    train_procedure_Shanghai = [(dataset,"train_data")]


    tf.compat.v1.reset_default_graph()


    x = tf.compat.v1.placeholder(dtype = tf.float32 ,  shape = [None , None , None , 3])
    GT = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [None , None , None , 3])
    Count = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [None , 1 ])
    

    
    score_scale = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [1])

    with tf.compat.v1.variable_scope("MODEL"):
        
        if model == "VGG19":
            MODEL = VGG19()
            
        elif model == "CSRNet":
            MODEL = CSRNet()
        
        elif model == "CAN":
            MODEL = CAN()

        
        z = MODEL.forward(x)

        z = tf.abs(z)
        

    loss , decoupled_density , expectation , variance = tools.Re_estimate(z , GT , score_scale)

    Saver = tf.compat.v1.train.Saver(max_to_keep = 20)
    
    initial = tf.compat.v1.global_variables_initializer()


    with tf.compat.v1.Session() as sess:

        sess.graph.finalize()
        
        sess.run(initial)
        
        Saver.restore(sess , "./model_last_epoch_for_inference_{0}/BLpp.ckpt".format(fold))
        
        print("\n[!] Model restoring...\n")


        for epoch in range(Epoch+1):
                         

            for (A_or_B ,  train_dir) in train_procedure_Shanghai:

                data_list = Shanghai.get_data_list(A_or_B = A_or_B , train_or_test = train_dir)

                data_list = data_list[fold * int(percentage * len(data_list)):(fold+1) * int(percentage * len(data_list))]


                print("[!] Re-estimate writing...")

                var_list = []
                for file in data_list:

                    (X_train_batch , Y_train_batch) = Shanghai.data_loader_pipeline(A_or_B = A_or_B , train_or_test = train_dir , 
                                                                                    data_list = [file] , write = True)
                
                    

                    dens , expects , var = sess.run([decoupled_density, expectation , variance],feed_dict={x:X_train_batch , 
                                                                                          GT:Y_train_batch[0][0] ,
                                                                                          Count:Y_train_batch[0][1],
                                                                                          score_scale:scale})

                    var_list.append(var[0])
               


                    _ = relabel_tools.rewrite_variance(A_or_B , file , dens , expects , var , True , Y_train_batch[0][0])
    
                
                var_array = np.concatenate(var_list , axis = 0)
                    
                print("[!] Re-estimate writing finished.")
                
                return var_array


def parse_args():

    parser = argparse.ArgumentParser(description='Preprocess ShanghaiTech raw data')

    parser.add_argument('--model' , default='VGG19',
                        help='[ VGG19 , CSRNet , CAN ] , you can only choose one of them')

    parser.add_argument('--data-dir', default='part_B_final',                        
                        help='data directory, input part_B_final or part_A_final to choose the data_dir')

    args = parser.parse_args()
    
    return args


if __name__ == "__main__" :
    
    args = parse_args()

    if args.data_dir not in ["part_B_final" , "part_A_final"]:
        assert False, "Please make sure that your argument about data_dir is part_B_final or part_A_final" 

    if args.model not in ["VGG19" , "CSRNet" , "CAN"]:
        assert False, "Please make sure that your argument about model is VGG19, CSRNet, or CAN"


    percentage = 0.2
   
    for i in range(5):
        
        _ = Training(fold = i , percentage = percentage , dataset =  args.data_dir , model = args.model)


    
