# Now we create our model
# Also remember that we would be using the Refine Net model 2 times.
# The green/blue bars in the diagram Fig 1 are the y after each operation/activation function
import tensorflow as tf
import scipy.io as sio
from keras.layers import Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import matplotlib.pyplot as py

# from google.colab import drive
# drive.mount('/content/drive')
# from google.colab import files
# uploaded_file = files.upload()

image_width = 32
image_length = 32
image_channel = 2  # Real Matrix and Imaginary Matrix
total_image = image_channel * image_length * image_width  # This is N in diagram
# We are using a 1/4 compress rate for our model
residual_dim = 2  # Number of time we want to use RefineNet in our model
encoded_dim = 512  # Compressed codeword M


def Neural_Network(x, residual, encoded_dim):
    def addcommonlayers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y
    
    def decoded(y):
        temp = y
        y = Conv2D(8, kernel_size=(3,3),padding = 'same' )(y)
        y = addcommonlayers(y)
        
        y = Conv2D(16, kernel_size = (3,3),padding = 'same')(y)
        y = addcommonlayers(y)

        y = Conv2D(2,kernel_size = (3,3),padding = 'same')(y)
        y = BatchNormalization()(y)

        y = add([temp,y])
        y = LeakyReLU()(y)

        return y
    
    x = Conv2D(2,(3,3),padding = 'same')(x)
    x = addcommonlayers(x)

    x = Reshape((total_image,))(x)
    encoded = Dense(encoded_dim, activation = 'linear')(x)

    x = Dense(total_image,activation='linear')(x)
    x = Reshape((image_length,image_width,image_channel))(x)

    for i in range(residual):
        x = decoded(x)

    x = Conv2D(2,(3,3),activation = 'sigmoid',padding = 'same')(x)

    return x
