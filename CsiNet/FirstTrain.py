import tensorflow as tf
import scipy.io as sio
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import matplotlib.pyplot as py
import numpy as np
import time
import math
from NeuralNetwork import Neural_Network, image_length, image_channel, image_width, residual_dim, encoded_dim
import NeuralNetwork as nn


image_width = 32
image_length = 32
image_channel = 2  # Real Matrix and Imaginary Matrix
total_image = image_channel * image_length * image_width  # This is N in diagram
# We are using a 1/4 compress rate for our model
residual_dim = 2  # Number of time we want to use RefineNet in our model
encoded_dim = 512  # Compressed codeword M


# Data loading
mat1 = sio.loadmat('DATA_Htrainin.mat')
x_train = mat1["HT"]  # array for accesing the HT variable in the matrix
mat2 = sio.loadmat('DATA_Hvalin.mat')
x_val = mat2["HT"]  # array for accesing the HT variable in the matrix
mat3 = sio.loadmat('DATA_Htestin.mat')
x_test = mat3["HT"]  # array for accesing the HT variable in the matrix.

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
# Preprocessing on the data.

# Now we typecast the array into a matrix using a reshape operation for the following file
# We need the elements to be in floating points of 32 bits.
x_train = x_train.astype(float)
x_val = x_val.astype(float)
x_test = x_test.astype(float)

# Giving the input in vector format so we have 4 dimensions to our data so reshape accordingly
x_train = np.reshape(
    x_train, (len(x_train), image_channel, image_length, image_width))
x_train = np.transpose(x_train, (0,2,3,1))
x_test = np.reshape(
    x_test, (len(x_test), image_channel, image_length, image_width))
x_test = np.transpose(x_test, (0,2,3,1))
x_val = np.reshape(
    x_val, (len(x_val), image_channel, image_length, image_width))
x_val = np.transpose(x_val, (0,2,3,1))


 
input_tensor = Input(shape=(image_length,image_width,image_channel))
output_value = Neural_Network(input_tensor, residual_dim, encoded_dim)
initial_lr=5e-3

# Now we create our model using the model function.
CsiNet = Model(inputs=[input_tensor], outputs=[output_value])
CsiNet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),loss='mse')
print(CsiNet.summary())



# Now we train our model to keep the performance of our model and keep track of our losses

lr_drop_period = 100
lr_drop_factor = 0.1

class LossHistory(Callback):
    # At the start of our training we intialize 2 variables to keep track of our losses
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []
    # We append to the end of the losses_train our current batch loss

    def on_batch_end(self,batch, logs={}):
        self.losses_train.append(logs.get('loss'))
    # We append to the end of the losses_val our validation loss

    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))
        
    def scheduler(epoch, lr):
        if lr_drop_period == np.Inf or np.mod(epoch,lr_drop_period) != 0:
            return lr
        else:
            return lr * tf.math.exp(-lr_drop_factor)


history = LossHistory()
# Create a file name
file = 'CsiNet_'+'_dim'+str(encoded_dim)+time.strftime('_%m_%d')
path = 'result/TensorBoard_%s' % file
earlyStopping = tf.keras.callbacks.EarlyStopping(patience=200,restore_best_weights=True)
#  We now train our model with our data with number of steps=1000, (samples per size =200) and shuffle data with each step
#  Also validate using x_val as both input and target
#  We also need to record the losses using callbacks
CsiNet.fit(x_train, x_train,
           epochs=100,
           batch_size=200,
           shuffle=True,
           validation_data=(x_val, x_val),
           callbacks=[history,
                      TensorBoard(log_dir=path), earlyStopping])

filename = 'result/trainloss_%s.csv' % file
loss_history = np.array(history.losses_train)
np.savetxt(filename, loss_history, delimiter=",")

filename = 'result/valloss_%s.csv' % file
loss_history = np.array(history.losses_val)
np.savetxt(filename, loss_history, delimiter=",")


# Now is the time to test our model and store in x_hat
tStart = time.time()
x_hat = CsiNet.predict(x_test)
tEnd = time.time()
print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

def NMSE(Hi, Hi_hat):
    Hi_real = np.reshape(Hi[:, : ,:, 0], (len(Hi), -1))
    Hi_img = np.reshape(Hi[:, :, :, 1], (len(Hi), -1))
    Hi_complex = (Hi_real-0.5)+1j*(Hi_img-0.5)

    Hi_hat_real = np.reshape(Hi_hat[:, :, :, 0], (len(Hi_hat), -1))
    Hi_hat_img = np.reshape(Hi_hat[:, :, :, 1], (len(Hi_hat), -1))
    Hi_hat_complex = (Hi_hat_real-0.5)+1j*(Hi_hat_img-0.5)

    mse = np.sum(np.square(np.abs(Hi_complex-Hi_hat_complex)), axis=1)
    power = np.sum(abs(Hi_complex)**2, axis=1)
    NMSE = 10* math.log10(np.mean(mse/power))
    return NMSE


print("NMSE is ", NMSE(x_test,x_hat))
filename = "result/decoded_%s.csv"%file
x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
np.savetxt(filename, x_hat1, delimiter=",")


# Serialize CSINET model to JSON
model_json = CsiNet.to_json()
outfile = "result/model_%s.json"%file
with open(outfile, "w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
outfile = "result/model_%s.h5"%file
CsiNet.save_weights(outfile)

# Save model in tensorflow format
CsiNet.save("tensorflow/model_%s"%file)


# (100000, 2048)
# (20000, 2048)
# (30000, 2048)
# WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
# WARNING:absl:There is a known slowdown when using v2.11+ Keras optimizers on M1/M2 Macs. Falling back to the legacy Keras optimizer, i.e., `tf.keras.optimizers.legacy.Adam`.
# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                Output Shape                 Param #   Connected to                  
# ==================================================================================================
#  input_1 (InputLayer)        [(None, 32, 32, 2)]          0         []                            
                                                                                                  
#  conv2d (Conv2D)             (None, 32, 32, 2)            38        ['input_1[0][0]']             
                                                                                                  
#  batch_normalization (Batch  (None, 32, 32, 2)            8         ['conv2d[0][0]']              
#  Normalization)                                                                                   
                                                                                                  
#  leaky_re_lu (LeakyReLU)     (None, 32, 32, 2)            0         ['batch_normalization[0][0]'] 
                                                                                                  
#  reshape (Reshape)           (None, 2048)                 0         ['leaky_re_lu[0][0]']         
                                                                                                  
#  dense_1 (Dense)             (None, 2048)                 4196352   ['reshape[0][0]']             
                                                                                                  
#  reshape_1 (Reshape)         (None, 32, 32, 2)            0         ['dense_1[0][0]']             
                                                                                                  
#  conv2d_1 (Conv2D)           (None, 32, 32, 8)            152       ['reshape_1[0][0]']           
                                                                                                  
#  batch_normalization_1 (Bat  (None, 32, 32, 8)            32        ['conv2d_1[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  leaky_re_lu_1 (LeakyReLU)   (None, 32, 32, 8)            0         ['batch_normalization_1[0][0]'
#                                                                     ]                             
                                                                                                  
#  conv2d_2 (Conv2D)           (None, 32, 32, 16)           1168      ['leaky_re_lu_1[0][0]']       
                                                                                                  
#  batch_normalization_2 (Bat  (None, 32, 32, 16)           64        ['conv2d_2[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  leaky_re_lu_2 (LeakyReLU)   (None, 32, 32, 16)           0         ['batch_normalization_2[0][0]'
#                                                                     ]                             
                                                                                                  
#  conv2d_3 (Conv2D)           (None, 32, 32, 2)            290       ['leaky_re_lu_2[0][0]']       
                                                                                                  
#  batch_normalization_3 (Bat  (None, 32, 32, 2)            8         ['conv2d_3[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  add (Add)                   (None, 32, 32, 2)            0         ['reshape_1[0][0]',           
#                                                                      'batch_normalization_3[0][0]'
#                                                                     ]                             
                                                                                                  
#  leaky_re_lu_3 (LeakyReLU)   (None, 32, 32, 2)            0         ['add[0][0]']                 
                                                                                                  
#  conv2d_4 (Conv2D)           (None, 32, 32, 8)            152       ['leaky_re_lu_3[0][0]']       
                                                                                                  
#  batch_normalization_4 (Bat  (None, 32, 32, 8)            32        ['conv2d_4[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  leaky_re_lu_4 (LeakyReLU)   (None, 32, 32, 8)            0         ['batch_normalization_4[0][0]'
#                                                                     ]                             
                                                                                                  
#  conv2d_5 (Conv2D)           (None, 32, 32, 16)           1168      ['leaky_re_lu_4[0][0]']       
                                                                                                  
#  batch_normalization_5 (Bat  (None, 32, 32, 16)           64        ['conv2d_5[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  leaky_re_lu_5 (LeakyReLU)   (None, 32, 32, 16)           0         ['batch_normalization_5[0][0]'
#                                                                     ]                             
                                                                                                  
#  conv2d_6 (Conv2D)           (None, 32, 32, 2)            290       ['leaky_re_lu_5[0][0]']       
                                                                                                  
#  batch_normalization_6 (Bat  (None, 32, 32, 2)            8         ['conv2d_6[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  add_1 (Add)                 (None, 32, 32, 2)            0         ['leaky_re_lu_3[0][0]',       
#                                                                      'batch_normalization_6[0][0]'
#                                                                     ]                             
                                                                                                  
#  leaky_re_lu_6 (LeakyReLU)   (None, 32, 32, 2)            0         ['add_1[0][0]']               
                                                                                                  
#  conv2d_7 (Conv2D)           (None, 32, 32, 2)            38        ['leaky_re_lu_6[0][0]']       
                                                                                                  
# ==================================================================================================
# Total params: 4199864 (16.02 MB)
# Trainable params: 4199756 (16.02 MB)
# Non-trainable params: 108 (432.00 Byte)
# __________________________________________________________________________________________________
# None
# Epoch 1/100
# 500/500 [==============================] - 84s 167ms/step - loss: 9.3461e-04 - val_loss: 3.8412e-04
# Epoch 2/100
# 500/500 [==============================] - 82s 164ms/step - loss: 2.3481e-04 - val_loss: 4.7048e-04
# Epoch 3/100
# 500/500 [==============================] - 81s 161ms/step - loss: 1.7585e-04 - val_loss: 6.3286e-04
# Epoch 4/100
# 500/500 [==============================] - 85s 170ms/step - loss: 1.4260e-04 - val_loss: 3.4737e-04
# Epoch 5/100
# 500/500 [==============================] - 81s 162ms/step - loss: 1.2185e-04 - val_loss: 3.1684e-04
# Epoch 6/100
# 500/500 [==============================] - 82s 164ms/step - loss: 1.0855e-04 - val_loss: 6.6984e-04
# Epoch 7/100
# 500/500 [==============================] - 82s 164ms/step - loss: 9.8857e-05 - val_loss: 3.9519e-04
# Epoch 8/100
# 500/500 [==============================] - 83s 166ms/step - loss: 9.5614e-05 - val_loss: 0.1115
# Epoch 9/100
# 500/500 [==============================] - 82s 164ms/step - loss: 9.1220e-05 - val_loss: 7.3461e-04
# Epoch 10/100
# 500/500 [==============================] - 82s 164ms/step - loss: 7.9615e-05 - val_loss: 8.2517e-05
# Epoch 11/100
# 500/500 [==============================] - 83s 165ms/step - loss: 7.6198e-05 - val_loss: 8.7695e-05
# Epoch 12/100
# 500/500 [==============================] - 83s 165ms/step - loss: 7.3212e-05 - val_loss: 7.3502e-05
# Epoch 13/100
# 500/500 [==============================] - 81s 161ms/step - loss: 7.2772e-05 - val_loss: 0.0057
# Epoch 14/100
# 500/500 [==============================] - 81s 161ms/step - loss: 6.7016e-05 - val_loss: 9.6272e-05
# Epoch 15/100
# 500/500 [==============================] - 80s 160ms/step - loss: 6.2500e-05 - val_loss: 1.0055e-04
# Epoch 16/100
# 500/500 [==============================] - 78s 157ms/step - loss: 6.1085e-05 - val_loss: 9.3010e-05
# Epoch 17/100
# 500/500 [==============================] - 78s 156ms/step - loss: 6.3871e-05 - val_loss: 0.0062
# Epoch 18/100
# 500/500 [==============================] - 78s 157ms/step - loss: 5.5829e-05 - val_loss: 2.6977e-04
# Epoch 19/100
# 500/500 [==============================] - 84s 169ms/step - loss: 5.4552e-05 - val_loss: 1.0841e-04
# Epoch 20/100
# 500/500 [==============================] - 83s 166ms/step - loss: 6.0032e-05 - val_loss: 0.0167
# Epoch 21/100
# 500/500 [==============================] - 83s 167ms/step - loss: 5.3092e-05 - val_loss: 9.0539e-04
# Epoch 22/100
# 500/500 [==============================] - 87s 174ms/step - loss: 4.9908e-05 - val_loss: 1.9339e-04
# Epoch 23/100
# 500/500 [==============================] - 80s 161ms/step - loss: 5.0628e-05 - val_loss: 9.3696e-05
# Epoch 24/100
# 500/500 [==============================] - 85s 170ms/step - loss: 4.8015e-05 - val_loss: 0.0151
# Epoch 25/100
# 500/500 [==============================] - 88s 175ms/step - loss: 4.9686e-05 - val_loss: 7.1486e-05
# Epoch 26/100
# 500/500 [==============================] - 86s 172ms/step - loss: 4.5898e-05 - val_loss: 3.4837e-04
# Epoch 27/100
# 500/500 [==============================] - 90s 180ms/step - loss: 4.6632e-05 - val_loss: 4.6288e-05
# Epoch 28/100
# 500/500 [==============================] - 83s 167ms/step - loss: 4.4127e-05 - val_loss: 9.2463e-05
# Epoch 29/100
# 500/500 [==============================] - 80s 161ms/step - loss: 4.3512e-05 - val_loss: 8.0812e-04
# Epoch 30/100
# 500/500 [==============================] - 81s 163ms/step - loss: 4.6696e-05 - val_loss: 4.7970e-05
# Epoch 31/100
# 500/500 [==============================] - 91s 183ms/step - loss: 4.1788e-05 - val_loss: 2.0004e-04
# Epoch 32/100
# 500/500 [==============================] - 81s 161ms/step - loss: 4.1948e-05 - val_loss: 1.1108e-04
# Epoch 33/100
# 500/500 [==============================] - 78s 156ms/step - loss: 3.9672e-05 - val_loss: 6.9539e-05
# Epoch 34/100
# 500/500 [==============================] - 78s 157ms/step - loss: 3.9412e-05 - val_loss: 2.5035e-04
# Epoch 35/100
# 500/500 [==============================] - 80s 160ms/step - loss: 3.7992e-05 - val_loss: 3.7059e-05
# Epoch 36/100
# 500/500 [==============================] - 81s 161ms/step - loss: 3.7424e-05 - val_loss: 4.3734e-05
# Epoch 37/100
# 500/500 [==============================] - 82s 164ms/step - loss: 3.7113e-05 - val_loss: 6.1376e-05
# Epoch 38/100
# 500/500 [==============================] - 83s 165ms/step - loss: 3.7217e-05 - val_loss: 5.8423e-05
# Epoch 39/100
# 500/500 [==============================] - 85s 171ms/step - loss: 3.6327e-05 - val_loss: 8.2713e-04
# Epoch 40/100
# 500/500 [==============================] - 85s 169ms/step - loss: 3.5402e-05 - val_loss: 1.2414e-04
# Epoch 41/100
# 500/500 [==============================] - 84s 169ms/step - loss: 3.4386e-05 - val_loss: 3.9415e-05
# Epoch 42/100
# 500/500 [==============================] - 78s 156ms/step - loss: 3.3684e-05 - val_loss: 1.6290e-04
# Epoch 43/100
# 500/500 [==============================] - 78s 156ms/step - loss: 3.4881e-05 - val_loss: 1.8229e-04
# Epoch 44/100
# 500/500 [==============================] - 78s 156ms/step - loss: 3.2858e-05 - val_loss: 1.5885e-04
# Epoch 45/100
# 500/500 [==============================] - 78s 156ms/step - loss: 3.2408e-05 - val_loss: 6.4842e-05
# Epoch 46/100
# 500/500 [==============================] - 79s 157ms/step - loss: 3.3594e-05 - val_loss: 0.0048
# Epoch 47/100
# 500/500 [==============================] - 79s 159ms/step - loss: 3.3229e-05 - val_loss: 1.5055e-04
# Epoch 48/100
# 500/500 [==============================] - 79s 158ms/step - loss: 3.0435e-05 - val_loss: 3.2194e-05
# Epoch 49/100
# 500/500 [==============================] - 79s 159ms/step - loss: 3.0507e-05 - val_loss: 3.0636e-05
# Epoch 50/100
# 500/500 [==============================] - 80s 159ms/step - loss: 3.0401e-05 - val_loss: 3.2442e-05
# Epoch 51/100
# 500/500 [==============================] - 79s 158ms/step - loss: 3.0069e-05 - val_loss: 7.5969e-05
# Epoch 52/100
# 500/500 [==============================] - 79s 159ms/step - loss: 3.0258e-05 - val_loss: 1.1841e-04
# Epoch 53/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.9031e-05 - val_loss: 2.9035e-05
# Epoch 54/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.8456e-05 - val_loss: 5.0194e-05
# Epoch 55/100
# 500/500 [==============================] - 79s 157ms/step - loss: 2.8514e-05 - val_loss: 2.9906e-05
# Epoch 56/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.8290e-05 - val_loss: 7.2592e-05
# Epoch 57/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.9666e-05 - val_loss: 1.6806e-04
# Epoch 58/100
# 500/500 [==============================] - 79s 157ms/step - loss: 2.7572e-05 - val_loss: 4.5053e-05
# Epoch 59/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.6744e-05 - val_loss: 3.8313e-05
# Epoch 60/100
# 500/500 [==============================] - 79s 159ms/step - loss: 2.6953e-05 - val_loss: 2.7172e-05
# Epoch 61/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.6366e-05 - val_loss: 3.4445e-05
# Epoch 62/100
# 500/500 [==============================] - 79s 157ms/step - loss: 2.5967e-05 - val_loss: 5.3766e-05
# Epoch 63/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.5701e-05 - val_loss: 2.6000e-05
# Epoch 64/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.5625e-05 - val_loss: 2.5068e-05
# Epoch 65/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.5075e-05 - val_loss: 2.3963e-05
# Epoch 66/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.4711e-05 - val_loss: 3.1564e-05
# Epoch 67/100
# 500/500 [==============================] - 79s 159ms/step - loss: 2.4920e-05 - val_loss: 2.6223e-05
# Epoch 68/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.4520e-05 - val_loss: 2.6263e-05
# Epoch 69/100
# 500/500 [==============================] - 82s 165ms/step - loss: 2.4419e-05 - val_loss: 9.3837e-05
# Epoch 70/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.3917e-05 - val_loss: 4.3423e-05
# Epoch 71/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.3633e-05 - val_loss: 2.4163e-05
# Epoch 72/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.3614e-05 - val_loss: 2.5530e-05
# Epoch 73/100
# 500/500 [==============================] - 78s 157ms/step - loss: 2.3316e-05 - val_loss: 2.5497e-05
# Epoch 74/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.3027e-05 - val_loss: 2.7979e-05
# Epoch 75/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.2737e-05 - val_loss: 4.3968e-05
# Epoch 76/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.2574e-05 - val_loss: 3.9803e-05
# Epoch 77/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.2913e-05 - val_loss: 3.0341e-05
# Epoch 78/100
# 500/500 [==============================] - 78s 157ms/step - loss: 2.2134e-05 - val_loss: 1.1422e-04
# Epoch 79/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.1723e-05 - val_loss: 2.2229e-05
# Epoch 80/100
# 500/500 [==============================] - 79s 158ms/step - loss: 2.2017e-05 - val_loss: 2.3940e-05
# Epoch 81/100
# 500/500 [==============================] - 78s 157ms/step - loss: 2.5010e-05 - val_loss: 9.1028e-04
# Epoch 82/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.1058e-05 - val_loss: 2.1224e-05
# Epoch 83/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.1051e-05 - val_loss: 4.2985e-05
# Epoch 84/100
# 500/500 [==============================] - 78s 155ms/step - loss: 2.1228e-05 - val_loss: 2.4414e-05
# Epoch 85/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.0694e-05 - val_loss: 6.0108e-05
# Epoch 86/100
# 500/500 [==============================] - 78s 156ms/step - loss: 2.0447e-05 - val_loss: 2.1391e-05
# Epoch 87/100
# 500/500 [==============================] - 82s 164ms/step - loss: 2.0572e-05 - val_loss: 2.6298e-05
# Epoch 88/100
# 500/500 [==============================] - 81s 162ms/step - loss: 2.0174e-05 - val_loss: 3.3653e-05
# Epoch 89/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.9974e-05 - val_loss: 1.9180e-05
# Epoch 90/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.9950e-05 - val_loss: 3.0227e-05
# Epoch 91/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.9722e-05 - val_loss: 4.3466e-05
# Epoch 92/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.9485e-05 - val_loss: 2.0076e-05
# Epoch 93/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.9248e-05 - val_loss: 2.5342e-05
# Epoch 94/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.9506e-05 - val_loss: 2.3511e-05
# Epoch 95/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.9336e-05 - val_loss: 1.8553e-05
# Epoch 96/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.9294e-05 - val_loss: 6.3741e-05
# Epoch 97/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.8948e-05 - val_loss: 1.1052e-04
# Epoch 98/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.8718e-05 - val_loss: 2.9059e-05
# Epoch 99/100
# 500/500 [==============================] - 77s 153ms/step - loss: 1.8742e-05 - val_loss: 3.0060e-05
# Epoch 100/100
# 500/500 [==============================] - 77s 154ms/step - loss: 1.8715e-05 - val_loss: 3.3960e-05
# 625/625 [==============================] - 5s 8ms/step
# It cost 0.000342 sec
# NMSE is  -11.007193875932487