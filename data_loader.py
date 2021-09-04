import scipy.io as scio
import numpy as np
import os
import os.path
import random
from PIL import Image



normalize = {"part_A_final" : ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898]),
             "part_B_final" : ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])
             }


def get_data_list(path = "." , A_or_B = None , train_or_test = None):

    image_path = os.path.join(path , A_or_B , train_or_test , "images")

    gt_path = os.path.join(path , A_or_B , train_or_test , "ground_truth_processed")
    
    file_list = [(i , j) for i , j in zip(os.listdir(image_path),os.listdir(gt_path))]
        
    return file_list
    


def data_loader_pipeline(Path = ".", A_or_B = None , train_or_test = None , data_list = None , write = False , training_stage = "1"):

    mean = np.array(normalize[A_or_B][0])[np.newaxis ,np.newaxis,:]
    std  = np.array(normalize[A_or_B][1])[np.newaxis ,np.newaxis,:]
   
         
    X = []
    Y = []
        
    for (x_file_name , y_file_name) in data_list:
            
        x_path = os.path.join(Path , A_or_B , train_or_test , "images" , x_file_name)
        y_path = os.path.join(Path , A_or_B , train_or_test , "ground_truth_processed" , y_file_name)
            
        image = np.array(Image.open(x_path)) / 255
            
        if len(image.shape) == 2:
                
            image = np.stack([image , image , image] , axis = 2 )
            image = (image - 0.5*np.ones([1,1,3])) / (0.5*np.ones([1,1,3]))
            
        else:
                
            image = (image - mean) / std

        data = scio.loadmat(y_path)
            
        dot = np.array(data["dot"])[np.newaxis,:,:,:]      
        count = np.array(data["total_count"])

        if training_stage == "1":
            sigma = 2
        else:
            sigma = data['sigma'][: , : , np.newaxis]
            
            
        if train_or_test == "train_data" and write == False:
                 
            if random.random() >= 0.5:
                
                image = np.flip(image , axis = 1)
                
                flip_value = np.array([image.shape[1]-1,0,0])[np.newaxis , np.newaxis , np.newaxis , :]
                
                keep_value = np.array([1,-1,-1])[np.newaxis , np.newaxis , np.newaxis , :]
                
                dot = flip_value - keep_value * dot
            
          
        resize_value = np.array([1/8,1/8,1])[np.newaxis , np.newaxis , np.newaxis , :]
            
        dot = dot * resize_value
                
        X.append(image)
        Y.append((dot,count,sigma))
        

    X = np.array(X)
    X = X[:,:,:,::-1]
        
    return ( X , Y )







if __name__ == "__main__" :

    print(0)




