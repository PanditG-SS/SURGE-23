import tensorflow as tf
from keras.layers import Conv3D,BatchNormalization,ReLU,Flatten,Reshape,MaxPooling3D, add ,Dense
from keras.models import Model
from math import floor
import numpy as np
#Now we store the dimensions of the input.

Nr=1
Nt=8
K=52

def Neural_Network(x,L):
    
    def add_Conv_Block(y,kernels,filter):
        y=Conv3D(filter,kernel_size=kernels,strides=(1,1,1),padding='same')(y)
        y=BatchNormalization()(y)
        y=ReLU()(y)
        return y
    
    def Conv_Res_Block(y):
        sidebranch=y;
        y= add_Conv_Block(y,(3,7,5),8*L)
        y= add_Conv_Block(y,(3,7,5),16*L)
        y=Conv3D( 4*L,kernel_size=(3,7,5),strides=(1,1,1),padding='same')(y)
        y=add([y,sidebranch])
        y=BatchNormalization()(y)
        y=ReLU()(y)
        return y
    
    def FC_Block(y):
        # X=2*floor(Nr)*floor(Nt/2)*floor(K/4)
        # # y=Reshape((X))(y) 
        y=Flatten()(y);
        y=Dense(2*Nr*Nt*K,activation='linear')(y)
        y=Reshape((2,Nr,Nt,K))(y)
        return y;
    
    x=add_Conv_Block(x,(3,7,5),4*L)
    x=MaxPooling3D(pool_size=(3,3,3),strides=(1,1,1),padding='same')(x)
    
    side=x
    
    for i in range (2):
        x=Conv_Res_Block(x)
        
    x=add([x,side])
    x=add_Conv_Block(x,(3,7,7),3)
    
    x=MaxPooling3D(pool_size=(4,1,2),strides=(4,1,2),padding='valid')(x)
    x=FC_Block(x)
    
    return x;
  
  
        
    
    
    
    
