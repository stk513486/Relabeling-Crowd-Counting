# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:45:37 2020

@author: a9090
"""



import tensorflow as tf
import numpy as np
import scipy.io as scio
import scipy
from PIL import Image


import time
import random
import os
import argparse

from models import VGG19 , CSRNet , CAN
import utils.tools_for_baseline as tools
import utils.relabel_tools as relabel_tools

import data_loader as Shanghai



def Training(fold = None , percentage = None , config = None , dataset = None, model = None):

    batch_SIZE = 1  # don't change it


    lr = 1e-5 
    Epoch = 130

    first_write_epoch = 50

    rewrite_epoch = 5

    first_write_flag = False  

    train_procedure = [(dataset,"train_data")]
    test_procedure = [(dataset,"test_data")]


    tf.compat.v1.reset_default_graph()


    x = tf.compat.v1.placeholder(dtype = tf.float32 ,  shape = [None , None , None , 3])
    GT = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [None , None , None , 3])
    Count = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [None , 1 ])
    
    LR = tf.compat.v1.placeholder(tf.float32)
    

    look_cost = tf.compat.v1.placeholder(tf.float32)
    look_MAE = tf.compat.v1.placeholder(tf.float32)
    look_MSE = tf.compat.v1.placeholder(tf.float32)

    with tf.compat.v1.variable_scope("MODEL"):

        if model == "VGG19":
            MODEL = VGG19()
            
        elif model == "CSRNet":
            MODEL = CSRNet()
        
        elif model == "CAN":
            MODEL = CAN()

        
        z = MODEL.forward(x)




    loss , decoupled_density , expectation = tools.ADSC_Loss(z , GT)


    all_variable = tf.compat.v1.trainable_variables()    
    L2_reg = 5e-6 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in all_variable ])

    
    performance = tools.compute_MAE_and_MSE(z , Count)


    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LR).minimize(loss , name = "Adam_All")
    weight_decay_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1.0).minimize(L2_reg , name = "Weight_decay")

    tf.compat.v1.summary.scalar("loss", look_cost)
    tf.compat.v1.summary.scalar("MAE" , look_MAE)
    tf.compat.v1.summary.scalar("MSE" , look_MSE)

    
    merged = tf.compat.v1.summary.merge_all()
    
    Saver = tf.compat.v1.train.Saver(max_to_keep = 20)
    
    initial = tf.compat.v1.global_variables_initializer()


    with tf.compat.v1.Session() as sess:

        print("-----------------------------------------------------------------------------\n")
        print("\nStart Training...\n")
        print("Number of parameters : " , np.sum([np.prod(v.get_shape().as_list()) for v in all_variable]) , "\n")


        
        Tk = 0
        seed = 860521
        best_MAE = 200
        best_MSE = 200
        f = open("logs_baseline_ADSCNet_{0}_{1}_{2}.txt".format(model , dataset[5] , str(fold)) , "a")
        
        
        train_writer = tf.compat.v1.summary.FileWriter("./logs/train",sess.graph)
        test_writer  = tf.compat.v1.summary.FileWriter("./logs/test" ,sess.graph)
    
        sess.graph.finalize()    
        sess.run(initial)



        random.seed(seed)
        
        data_list = Shanghai.get_data_list(A_or_B = train_procedure[0][0] , train_or_test = train_procedure[0][1])
        
        data_list = data_list[fold * int(percentage * len(data_list)):(fold+1) * int(percentage * len(data_list))]

        
        for epoch in range(Epoch+1):
            
            if epoch == 40:
                lr = lr / 2
            if epoch == 80:
                lr = lr / 2
            if epoch == 120:
                lr = lr / 2
                
            start_time = time.time()
        
            mini_batch_cost = 0
            mini_batch_MAE = 0
            mini_batch_MSE = 0
        
            seed = seed + 1
            

            #-----------------------------------------------------------------------------------------------
            # Train
            train_num , test_num = 0 , 0 
            

            for (A_or_B ,  train_dir) in train_procedure:
                
                random.seed(seed)
                random.shuffle(data_list)

                
                
                if len(data_list) % batch_SIZE == 0:
                    number_of_batch = int(len(data_list) / batch_SIZE)
                else:
                    number_of_batch = int(len(data_list) / batch_SIZE) + 1

                for r in range(number_of_batch):
                    
                
                    (X_train_batch , Y_train_batch) = Shanghai.data_loader_pipeline(A_or_B = A_or_B , train_or_test = train_dir , 
                                                                                    data_list = data_list[r*batch_SIZE:(r+1)*batch_SIZE] )
                
                    

                    _ , temp_cost ,train_performance = sess.run([optimizer , loss , performance],feed_dict={x:X_train_batch , 
                                                                                                           GT:Y_train_batch[0][0] ,
                                                                                                           Count:Y_train_batch[0][1],
                                                                                                           LR:lr
                                                                                                           })
                    _ = sess.run([weight_decay_op])
    

    
                    mini_batch_cost += temp_cost * X_train_batch.shape[0]                
                    mini_batch_MAE  += train_performance[0]
                    mini_batch_MSE  += train_performance[1]
                
                    train_num += X_train_batch.shape[0]
                
                

            
            total_cost = round(mini_batch_cost / train_num , 7)
            total_MAE  = round(mini_batch_MAE / train_num , 4)
            total_MSE  = round(np.sqrt(mini_batch_MSE / train_num) , 4)
          
            print("Shanghai : ",epoch , " , Cost :  " , total_cost , " , MAE : ",  total_MAE , ", MSE : " , total_MSE)
            f.write("Shanghai : " + str(epoch) + " , Cost :  " + str(total_cost) +" , MAE : "+str(total_MAE)+" , MSE : "+str(total_MSE) +"\n")


            result = sess.run(merged , feed_dict={look_cost:total_cost, 
                                                  look_MAE :total_MAE,
                                                  look_MSE :total_MSE})

    
            train_writer.add_summary(result , epoch)            

            #---------------------------------------------------------------------------------------------------------------
            # rewrite
            if epoch == first_write_epoch and first_write_flag == False and config[0] == True:
                
                first_write_flag = True
                print("[!] First writing...")

                for file in data_list:

                    (X_train_batch , Y_train_batch) = Shanghai.data_loader_pipeline(A_or_B = A_or_B , train_or_test = train_dir , 
                                                                                    data_list = [file] , write = True)
                
                    

                    dens , expects = sess.run([decoupled_density, expectation],feed_dict={x:X_train_batch , 
                                                                                          GT:Y_train_batch[0][0] ,
                                                                                          Count:Y_train_batch[0][1]})



                    relabel_tools.relabel_operation(A_or_B , file , dens , expects , True , Y_train_batch[0][0])
                
                print("[!] First writing finished.")
                
    
            elif epoch % rewrite_epoch == 0 and first_write_flag == True and config[0] == True:
                
                print("[i] Re-label writing..." )
                for file in data_list:
                    
                    (X_train_batch , Y_train_batch) = Shanghai.data_loader_pipeline(A_or_B = A_or_B , train_or_test = train_dir , 
                                                                                    data_list = [file] , write = True)
                
                    

                    dens , expects = sess.run([decoupled_density, expectation],feed_dict={x:X_train_batch , 
                                                                                          GT:Y_train_batch[0][0] ,
                                                                                          Count:Y_train_batch[0][1]})
      
    
                    if True:
                        
                        relabel_tools.relabel_operation(A_or_B , file , dens , expects , False , Y_train_batch[0][0] , 1)
                
                print("[i] Re-label finished." )
        
                
            else:
                pass

            #--------------------------------------------------------------------------------
            # Test

            if epoch >= (Epoch//2) and epoch % 2 == 0:
             
                test_cost  , test_MAE , test_MSE = 0,0,0

                for (A_or_B ,  test_dir) in test_procedure:
                    

                    test_data_list = Shanghai.get_data_list(A_or_B = A_or_B ,train_or_test = test_dir)
                
                    for single_data in test_data_list:
                    
                        (X_test_batch , Y_test_batch ) = Shanghai.data_loader_pipeline( A_or_B = A_or_B  , train_or_test = test_dir,
                                                                                       data_list = [single_data])
        
                        temp_cost , test_performance  = sess.run([loss , performance ] , feed_dict={ x:X_test_batch , 
                                                                                                     GT:Y_test_batch[0][0],
                                                                                                     Count:Y_test_batch[0][1]})
        

                        test_cost += temp_cost
                        test_MAE += test_performance[0]
                        test_MSE += test_performance[1]
                    
                        test_num += 1


                test_cost = round(test_cost / test_num , 7)
                test_MAE = round(test_MAE / test_num , 4)
                test_MSE = round(np.sqrt(test_MSE / test_num ) , 4)
          
                print("Testing , cost :  " , test_cost , " , MAE : ", test_MAE," , MSE : " , test_MSE)
                f.write("Testing , cost :  " +str( test_cost) + " , MAE : " +str( test_MAE) + " , MSE : " +str( test_MSE)+"\n")



                result = sess.run(merged , feed_dict={look_cost:test_cost, 
                                                      look_MAE :test_MAE,
                                                      look_MSE :test_MSE})

    
                test_writer.add_summary(result , epoch)            

                if test_MAE <= best_MAE:
                    best_MAE = test_MAE
                    best_MSE = test_MSE
                    Saver.save(sess,"./model_baseline_ADSCNet_{0}_{1}_{2}/BLpp.ckpt".format(model , dataset[5] , str(fold)))


            #------------------------------------------------------------------------------------------------  
             
            process_time = time.time() - start_time        
            Tk = Tk + (process_time - Tk) / (epoch + 1 )

            if epoch % 10 == 0 :
            
                print("\nAverage training time  per epoch : " , Tk,)
                print("Best MAE : {0} , Best MSE : {1}\n".format(best_MAE,best_MSE))

            if epoch == Epoch:

                f.write("MAE : " +str(best_MAE) + " , MSE : " +str(best_MSE)+"\n")


        f.close()

    print("\nDone\n")

    return 





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
    
    config = [True] 
    # turn on the relabel mechanism
    
    for i in range(5):

        if not os.path.isdir(".\\model_baseline_ADSCNet_{0}_{1}_{2}".format(args.model , args.data_dir[5], str(i))):
            
            os.makedirs(".\\model_baseline_ADSCNet_{0}_{1}_{2}".format(args.model , args.data_dir[5], str(i)))

        Training(fold = i , percentage = percentage , config = config , dataset =  args.data_dir , model = args.model)


