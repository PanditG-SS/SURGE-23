import numpy as np
import tensorflow as tf
# import scipy.io as sio
from keras.layers import Input
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import matplotlib.pyplot as py
import numpy as np
import mat73
import time
import math

from NeuralNetwork import Neural_Network
batch_sizer = 512
Nr = 1
Nt = 8
K = 52


# Data loading
mat1 = mat73.loadmat('test_prediction.mat')
x_test = mat1["HT"]  # array for accesing the HT variable in the matrix
mat2 = mat73.loadmat('train_prediction.mat')
x_train = mat2["HT"]  # array for accesing the HT variable in the matrix

L = 3


x_train = x_train.astype(float)
x_test = x_test.astype(float)


print(x_train.shape)
print(x_test.shape)
# # (85779, 8, 8, 52)
# # (22222, 8, 8, 52)


x_test = np.reshape(x_test, (len(x_test), 2*(L+1), Nr, Nt, K))
x_train = np.reshape(x_train, (len(x_train), 2*(L+1), Nr, Nt, K))

input_test_samples = np.concatenate((x_test[:, 0:3], x_test[:, 4:7]), axis=1)
input_train_samples = np.concatenate(
    (x_train[:, 0:3], x_train[:, 4:7]), axis=1)
output_test_samples = np.concatenate((x_test[:, 3:4], x_test[:, 7:8]), axis=1)
output_train_samples = np.concatenate(
    (x_train[:, 3:4], x_train[:, 7:8]), axis=1)


input_test_samples = np.reshape(input_test_samples, (-1, 2*L, Nr, Nt, K))
input_train_samples = np.reshape(input_train_samples, (-1, 2*L, Nr, Nt, K))
output_test_samples = np.reshape(output_test_samples, (-1, 2, Nr, Nt, K))
output_train_samples = np.reshape(output_train_samples, (-1, 2, Nr, Nt, K))


print(input_train_samples.shape)
print(input_test_samples.shape)
print(output_train_samples.shape)
print(output_test_samples.shape)

# (85779, 6, 1, 8, 52)
# (22222, 6, 1, 8, 52)
# (85779, 2, 1, 8, 52)
# (22222, 2, 1, 8, 52)


input_tensor = Input(shape=(2*L, Nr, Nt, K))
output_value = Neural_Network(input_tensor, L)

initial_lr = 1e-3

CsiNet = Model(inputs=[input_tensor], outputs=[output_value])
CsiNet.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=initial_lr), loss='mse')
print(CsiNet.summary())

lr_drop_period1 = 100
lr_drop_period2 = 200
lr_drop_period3 = 250
lr_drop_factor = 0.1


class LossHistory(Callback):
    # At the start of our training we intialize 2 variables to keep track of our losses
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []
    # We append to the end of the losses_train our current batch loss

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
    # We append to the end of the losses_val our validation loss

    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))

    def scheduler(epoch, lr):
        if lr_drop_period1 == np.Inf or np.mod(epoch, lr_drop_period1) != 0 or lr_drop_period2 == np.Inf or np.mod(epoch, lr_drop_period2) != 0 or lr_drop_period3 == np.Inf or np.mod(epoch, lr_drop_period3) != 0:
            return lr
        else:
            return lr * tf.math.exp(-lr_drop_factor)


history = LossHistory()
# Create a file name
file = 'CsiNet_'+'_dim'+time.strftime('_%m_%d')
path = 'result/TensorBoard_%s' % file
earlyStopping = tf.keras.callbacks.EarlyStopping(
    patience=200, restore_best_weights=True)
#  We now train our model with our data with number of steps=1000, (samples per size =200) and shuffle data with each step
#  Also validate using x_val as both input and target
#  We also need to record the losses using callbacks
CsiNet.fit(input_train_samples, output_train_samples,
           epochs=50,
           batch_size=512,
           validation_data=(input_train_samples, output_train_samples),
           shuffle=True,
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
x_hat = CsiNet.predict(input_test_samples)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart)/input_test_samples.shape[0]))



def NMSE(Hi, Hi_hat):
    Hi_real = np.reshape(Hi[:, 0, :, :, :], (len(Hi), -1))
    Hi_img = np.reshape(Hi[:, 1, :, :, :], (len(Hi), -1))
    Hi_complex = (Hi_real-0.5)+1j*(Hi_img-0.5)

    Hi_hat_real = np.reshape(Hi_hat[:, 0, :, :, :], (len(Hi_hat), -1))
    Hi_hat_img = np.reshape(Hi_hat[:, 1, :, :, :], (len(Hi_hat), -1))
    Hi_hat_complex = (Hi_hat_real-0.5)+1j*(Hi_hat_img-0.5)

    mse = np.sum(np.square(np.abs(Hi_complex - Hi_hat_complex)), axis=(1))
    power = np.sum(np.abs(Hi_complex) ** 2, axis=(1))
    NMSE = 10 * np.log10(np.mean(mse) / np.maximum(np.mean(power), 1e-10))
    return NMSE



print(x_hat.shape)
print(output_test_samples.shape)

print("NMSE is ", NMSE(output_test_samples, x_hat))
filename = "result/decoded_%s.csv" % file
x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
np.savetxt(filename, x_hat1, delimiter=",")

# Serialize CSINET model to JSON
model_json = CsiNet.to_json()
outfile = "result/model_%s.json" % file
with open(outfile, "w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
outfile = "result/model_%s.h5" % file
CsiNet.save_weights(outfile)

# Save model in tensorflow format
CsiNet.save("tensorflow/model_%s" % file)

# /usr/local/bin/python3 "/Users/shobhitsharma/Desktop/INTERN/ML & AI/MODEL 2/Trainer.py"
# shobhitsharma@Shobhits-MacBook-Air MODEL 2 % /usr/local/bin/python3 "/Users/shobhitsharma/Desktop/INTERN/M
# L & AI/MODEL 2/Trainer.py"
# (85779, 8, 8, 52)
# (22222, 8, 8, 52)
# (85779, 6, 1, 8, 52)
# (22222, 6, 1, 8, 52)
# (85779, 2, 1, 8, 52)
# (22222, 2, 1, 8, 52)
# WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
# WARNING:absl:There is a known slowdown when using v2.11+ Keras optimizers on M1/M2 Macs. Falling back to the legacy Keras optimizer, i.e., `tf.keras.optimizers.legacy.Adam`.
# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                Output Shape                 Param #   Connected to                  
# ==================================================================================================
#  input_1 (InputLayer)        [(None, 6, 1, 8, 52)]        0         []                            
                                                                                                  
#  conv3d (Conv3D)             (None, 6, 1, 8, 12)          16860     ['input_1[0][0]']             
                                                                                                  
#  batch_normalization (Batch  (None, 6, 1, 8, 12)          48        ['conv3d[0][0]']              
#  Normalization)                                                                                   
                                                                                                  
#  re_lu (ReLU)                (None, 6, 1, 8, 12)          0         ['batch_normalization[0][0]'] 
                                                                                                  
#  max_pooling3d (MaxPooling3  (None, 6, 1, 8, 12)          0         ['re_lu[0][0]']               
#  D)                                                                                               
                                                                                                  
#  conv3d_1 (Conv3D)           (None, 6, 1, 8, 24)          30264     ['max_pooling3d[0][0]']       
                                                                                                  
#  batch_normalization_1 (Bat  (None, 6, 1, 8, 24)          96        ['conv3d_1[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  re_lu_1 (ReLU)              (None, 6, 1, 8, 24)          0         ['batch_normalization_1[0][0]'
#                                                                     ]                             
                                                                                                  
#  conv3d_2 (Conv3D)           (None, 6, 1, 8, 48)          121008    ['re_lu_1[0][0]']             
                                                                                                  
#  batch_normalization_2 (Bat  (None, 6, 1, 8, 48)          192       ['conv3d_2[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  re_lu_2 (ReLU)              (None, 6, 1, 8, 48)          0         ['batch_normalization_2[0][0]'
#                                                                     ]                             
                                                                                                  
#  conv3d_3 (Conv3D)           (None, 6, 1, 8, 12)          60492     ['re_lu_2[0][0]']             
                                                                                                  
#  add (Add)                   (None, 6, 1, 8, 12)          0         ['conv3d_3[0][0]',            
#                                                                      'max_pooling3d[0][0]']       
                                                                                                  
#  batch_normalization_3 (Bat  (None, 6, 1, 8, 12)          48        ['add[0][0]']                 
#  chNormalization)                                                                                 
                                                                                                  
#  re_lu_3 (ReLU)              (None, 6, 1, 8, 12)          0         ['batch_normalization_3[0][0]'
#                                                                     ]                             
                                                                                                  
#  conv3d_4 (Conv3D)           (None, 6, 1, 8, 24)          30264     ['re_lu_3[0][0]']             
                                                                                                  
#  batch_normalization_4 (Bat  (None, 6, 1, 8, 24)          96        ['conv3d_4[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  re_lu_4 (ReLU)              (None, 6, 1, 8, 24)          0         ['batch_normalization_4[0][0]'
#                                                                     ]                             
                                                                                                  
#  conv3d_5 (Conv3D)           (None, 6, 1, 8, 48)          121008    ['re_lu_4[0][0]']             
                                                                                                  
#  batch_normalization_5 (Bat  (None, 6, 1, 8, 48)          192       ['conv3d_5[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  re_lu_5 (ReLU)              (None, 6, 1, 8, 48)          0         ['batch_normalization_5[0][0]'
#                                                                     ]                             
                                                                                                  
#  conv3d_6 (Conv3D)           (None, 6, 1, 8, 12)          60492     ['re_lu_5[0][0]']             
                                                                                                  
#  add_1 (Add)                 (None, 6, 1, 8, 12)          0         ['conv3d_6[0][0]',            
#                                                                      're_lu_3[0][0]']             
                                                                                                  
#  batch_normalization_6 (Bat  (None, 6, 1, 8, 12)          48        ['add_1[0][0]']               
#  chNormalization)                                                                                 
                                                                                                  
#  re_lu_6 (ReLU)              (None, 6, 1, 8, 12)          0         ['batch_normalization_6[0][0]'
#                                                                     ]                             
                                                                                                  
#  add_2 (Add)                 (None, 6, 1, 8, 12)          0         ['re_lu_6[0][0]',             
#                                                                      'max_pooling3d[0][0]']       
                                                                                                  
#  conv3d_7 (Conv3D)           (None, 6, 1, 8, 3)           5295      ['add_2[0][0]']               
                                                                                                  
#  batch_normalization_7 (Bat  (None, 6, 1, 8, 3)           12        ['conv3d_7[0][0]']            
#  chNormalization)                                                                                 
                                                                                                  
#  re_lu_7 (ReLU)              (None, 6, 1, 8, 3)           0         ['batch_normalization_7[0][0]'
#                                                                     ]                             
                                                                                                  
#  max_pooling3d_1 (MaxPoolin  (None, 1, 1, 4, 3)           0         ['re_lu_7[0][0]']             
#  g3D)                                                                                             
                                                                                                  
#  flatten (Flatten)           (None, 12)                   0         ['max_pooling3d_1[0][0]']     
                                                                                                  
#  dense (Dense)               (None, 832)                  10816     ['flatten[0][0]']             
                                                                                                  
#  reshape (Reshape)           (None, 2, 1, 8, 52)          0         ['dense[0][0]']               
                                                                                                  
# ==================================================================================================
# Total params: 457231 (1.74 MB)
# Trainable params: 456865 (1.74 MB)
# Non-trainable params: 366 (1.43 KB)
# __________________________________________________________________________________________________
# None
# Epoch 1/50
# 168/168 [==============================] - 114s 678ms/step - loss: 0.0115 - val_loss: 0.0109
# Epoch 2/50
# 168/168 [==============================] - 113s 675ms/step - loss: 0.0104 - val_loss: 0.0106
# Epoch 3/50
# 168/168 [==============================] - 113s 671ms/step - loss: 0.0101 - val_loss: 0.0104
# Epoch 4/50
# 168/168 [==============================] - 113s 673ms/step - loss: 0.0098 - val_loss: 0.0097
# Epoch 5/50
# 168/168 [==============================] - 114s 676ms/step - loss: 0.0091 - val_loss: 0.0091
# Epoch 6/50
# 168/168 [==============================] - 114s 680ms/step - loss: 0.0085 - val_loss: 0.0085
# Epoch 7/50
# 168/168 [==============================] - 114s 677ms/step - loss: 0.0083 - val_loss: 0.0082
# Epoch 8/50
# 168/168 [==============================] - 113s 674ms/step - loss: 0.0080 - val_loss: 0.0079
# Epoch 9/50
# 168/168 [==============================] - 113s 676ms/step - loss: 0.0076 - val_loss: 0.0075
# Epoch 10/50
# 168/168 [==============================] - 113s 675ms/step - loss: 0.0073 - val_loss: 0.0072
# Epoch 11/50
# 168/168 [==============================] - 114s 676ms/step - loss: 0.0071 - val_loss: 0.0070
# Epoch 12/50
# 168/168 [==============================] - 113s 674ms/step - loss: 0.0069 - val_loss: 0.0070
# Epoch 13/50
# 168/168 [==============================] - 114s 677ms/step - loss: 0.0068 - val_loss: 0.0068
# Epoch 14/50
# 168/168 [==============================] - 113s 676ms/step - loss: 0.0067 - val_loss: 0.0068
# Epoch 15/50
# 168/168 [==============================] - 113s 674ms/step - loss: 0.0067 - val_loss: 0.0067
# Epoch 16/50
# 168/168 [==============================] - 114s 678ms/step - loss: 0.0067 - val_loss: 0.0067
# Epoch 17/50
# 168/168 [==============================] - 116s 694ms/step - loss: 0.0066 - val_loss: 0.0067
# Epoch 18/50
# 168/168 [==============================] - 113s 676ms/step - loss: 0.0066 - val_loss: 0.0066
# Epoch 19/50
# 168/168 [==============================] - 114s 678ms/step - loss: 0.0066 - val_loss: 0.0066
# Epoch 20/50
# 168/168 [==============================] - 115s 683ms/step - loss: 0.0066 - val_loss: 0.0066
# Epoch 21/50
# 168/168 [==============================] - 118s 700ms/step - loss: 0.0066 - val_loss: 0.0066
# Epoch 22/50
# 168/168 [==============================] - 115s 684ms/step - loss: 0.0065 - val_loss: 0.0066
# Epoch 23/50
# 168/168 [==============================] - 114s 677ms/step - loss: 0.0065 - val_loss: 0.0065
# Epoch 24/50
# 168/168 [==============================] - 115s 687ms/step - loss: 0.0065 - val_loss: 0.0065
# Epoch 25/50
# 168/168 [==============================] - 117s 699ms/step - loss: 0.0065 - val_loss: 0.0065
# Epoch 26/50
# 168/168 [==============================] - 114s 680ms/step - loss: 0.0065 - val_loss: 0.0065
# Epoch 27/50
# 168/168 [==============================] - 114s 677ms/step - loss: 0.0065 - val_loss: 0.0065
# Epoch 28/50
# 168/168 [==============================] - 113s 672ms/step - loss: 0.0065 - val_loss: 0.0065
# Epoch 29/50
# 168/168 [==============================] - 113s 671ms/step - loss: 0.0065 - val_loss: 0.0065
# Epoch 30/50
# 168/168 [==============================] - 112s 669ms/step - loss: 0.0065 - val_loss: 0.0065
# Epoch 31/50
# 168/168 [==============================] - 112s 668ms/step - loss: 0.0065 - val_loss: 0.0064
# Epoch 32/50
# 168/168 [==============================] - 113s 672ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 33/50
# 168/168 [==============================] - 112s 670ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 34/50
# 168/168 [==============================] - 113s 670ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 35/50
# 168/168 [==============================] - 113s 671ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 36/50
# 168/168 [==============================] - 112s 670ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 37/50
# 168/168 [==============================] - 112s 670ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 38/50
# 168/168 [==============================] - 113s 671ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 39/50
# 168/168 [==============================] - 113s 672ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 40/50
# 168/168 [==============================] - 112s 670ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 41/50
# 168/168 [==============================] - 112s 668ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 42/50
# 168/168 [==============================] - 112s 667ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 43/50
# 168/168 [==============================] - 112s 668ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 44/50
# 168/168 [==============================] - 112s 668ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 45/50
# 168/168 [==============================] - 113s 672ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 46/50
# 168/168 [==============================] - 114s 679ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 47/50
# 168/168 [==============================] - 112s 667ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 48/50
# 168/168 [==============================] - 112s 669ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 49/50
# 168/168 [==============================] - 112s 670ms/step - loss: 0.0064 - val_loss: 0.0064
# Epoch 50/50
# 168/168 [==============================] - 112s 667ms/step - loss: 0.0064 - val_loss: 0.0064
# 695/695 [==============================] - 8s 11ms/step
# It cost 0.000475 sec
# (22222, 2, 1, 8, 52)
# (22222, 2, 1, 8, 52)
# NMSE is  -16.05771959693543
# shobhitsharma@Shobhits-MacBook-Air MODEL 2 % 