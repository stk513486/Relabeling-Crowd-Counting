# -*- coding: utf-8 -*-


import numpy as np
from PIL import Image
import scipy.io as scio
import scipy

import os
import os.path
import time

import argparse



def preprocess_data(A_or_B = "part_B_final" , train_or_test = "train_data"):
    
    path = "."
    
    path_gt = os.path.join(path , A_or_B , train_or_test , "ground_truth")
    
    path_gt_processed = os.path.join(path , A_or_B , train_or_test , "ground_truth_processed")
    
    path_image = os.path.join(path , A_or_B , train_or_test , "images")

    
    if not os.path.exists(path_gt_processed):
        os.makedirs(path_gt_processed)    
    
    file_list = [name for name in os.listdir(path_gt)]
    

    for i in file_list:
        
        image_file_name = i[3:-4] + ".jpg"
        
        data = scio.loadmat(os.path.join(path_gt , str(i)))
        
        image = Image.open(os.path.join(path_image , image_file_name))
        shape = np.array(image).shape[:2]
        
        data = data["image_info"][0][0]['location'][0][0]

        print(i)
        
        get_density_map_and_save(data , shape , path_gt_processed , i , A_or_B)





def get_density_map_and_save(data = None , shape = None , path_gt_processed = None , name = None , A_or_B = None):
    
    dot = []
     
    if True:

        for index in range(data.shape[0]):
        
            x = data[index][0]
            y = data[index][1]
            
            if x >= shape[1]:
                x = shape[1] - 1
            elif x <= 0 :
                x = 0
            
            if y >= shape[0]:
                y = shape[0] - 1
            elif y <= 0:
                y = 0            
                 
            
            coefficient = 1
            dot.append((x , y , coefficient))
            
        
        dot = np.array(dot)[:,np.newaxis,:]
        sigma = 2.0 * np.ones(shape = [data.shape[0]])
        label = {
                 "total_count" : data.shape[0],
                 "dot" : dot,
                 "sigma" : sigma
                 }

        assert data.shape[0] == len(dot) , "Number of dots doesn't equal to original data"
        
        scio.savemat(os.path.join(path_gt_processed , name) , label)


    return 0


def parse_args():

    parser = argparse.ArgumentParser(description='Preprocess ShanghaiTech raw data')


    parser.add_argument('--data-dir', default='part_B_final',                        
                        help='data directory, input part_B_final, part_A_final or both_A_B to choose the data_dir')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    

    st = time.time()

    args = parse_args()
    
    if args.data_dir not in ["part_A_final" , "part_B_final" , "both_A_B"]:
        assert False, "Please make sure that your argument about data-dir is part_A_final, part_B_final, or both_A_B."
        
        
    if args.data_dir == "both_A_B":

        preprocess_data("part_A_final" , "train_data")

        preprocess_data("part_A_final" , "test_data")

        preprocess_data("part_B_final" , "train_data")

        preprocess_data("part_B_final" , "test_data")
    
    else:
        
        preprocess_data(args.data_dir , "train_data")

        preprocess_data(args.data_dir , "test_data")


    print("Done , the process time : {0}\n".format(time.time()-st))
    

