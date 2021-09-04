# -*- coding: utf-8 -*-

import numpy as np
import os
import scipy.io as scio
import scipy




def relabel_operation(A_or_B = None , file = None , density = None , expectation = None , first_write = False , orig_GT = None , keep_dots = 5):
    
    path = os.path.join(os.getcwd() , A_or_B , "train_data" , "ground_truth_processed")
    
    if first_write == True:
        
        dot = []
    
        for i in range(expectation.shape[1]):
            
            x = 8.0 * orig_GT[0][i][0][0]
            y = 8.0 * orig_GT[0][i][0][1]
            
            coefficient = np.abs(density[0][i] - 1)
            
            dot.append([x , y , coefficient])
            
        assert expectation.shape[1] == orig_GT.shape[1] , "Number of dot doesn't equal to data"

        dot = np.array(dot)[:,np.newaxis,:]
           
        label = {
                 "total_count" : expectation.shape[1],
                 "dot" : dot                
                }
        
        scio.savemat(os.path.join(path , file[1]) , label)


    else:

        k = keep_dots
        
        new_dot = []

        for i in range(expectation.shape[1]):
            
            x = 8.0 * expectation[0][i][0]
            y = 8.0 * expectation[0][i][1]
            
            coefficient = np.abs(density[0][i] - 1)
            
            
            new_dot.append([x , y , coefficient])
        
        new_dot = np.array(new_dot)[np.newaxis , : , np.newaxis , :]
        
        keepsize = np.array([8 , 8 , 1])[np.newaxis , np.newaxis , np.newaxis , :]
        
        orig_GT = keepsize * orig_GT
        
        dot = np.concatenate([orig_GT , new_dot] , axis = 2)[0]

        ##        
        score = dot[:,:,2]
        get = np.argsort(score , axis = 1)
        get = get[:,:k]
        dot = dot[np.indices(get.shape)[0],get,:]
        ##
        
        
        assert expectation.shape[1] == orig_GT.shape[1] , "Number of dot doesn't equal to data"
        
        assert len(dot.shape) == 3 , "Shape of dot ERROR"
        
        label = {
                 "total_count" : expectation.shape[1],
                 "dot" : dot
                }
        
        scio.savemat(os.path.join(path , file[1]) , label)
        
        

    return 0





def rewrite_variance(A_or_B = None , file = None , density = None , expectation = None , var = None , first_write = False , orig_GT = None):
    
    path = os.path.join(os.getcwd() , A_or_B , "train_data" , "ground_truth_processed")
    
    if first_write == True:
        
    
        data = scio.loadmat(os.path.join(path , file[1]))
        
        sigma = 0.5*(var[0,:,0]+var[0,:,1])
        sigma = np.sqrt(sigma)

        print(sigma.shape)
           
        label = {
                 "total_count" : data['total_count'],
                 "dot" : data['dot'],
                 "sigma" : sigma
                }
        
        scio.savemat(os.path.join(path , file[1]) , label)




    return 0

