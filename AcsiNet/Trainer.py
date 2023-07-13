import numpy as np
import tensorflow as tf
import scipy.io as sio
from keras.layers import Input
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import matplotlib.pyplot as py

# !pip install mat73
import mat73
import numpy as np
import time
import math

# from NeuralNetwork import NeuralNetwork
# Now we store the dimensions of the input.
channel = 2

M = 26
Nx = 8
Ny = 4

r = 16

import tensorflow as tf
from keras.layers import Conv3D, add, BatchNormalization, LeakyReLU, ReLU, Flatten, Reshape, GlobalMaxPooling3D, MaxPooling3D, multiply, Concatenate, GlobalAveragePooling3D, AveragePooling3D, AveragePooling3D, Dense
from keras.models import Model
from math import floor
import numpy as np
# Now we store the dimensions of the input.



# We define Channel attention module


def channel_attention_module(F, r, channel):

    out1 = MaxPooling3D(pool_size=(Ny, M, Nx), data_format='channels_last')(F)
    out2 = AveragePooling3D(pool_size=(Ny, M, Nx),
                            data_format='channels_last')(F)

    def MLP(Fc, r, channel):
        ans1 = Dense(units=channel/r, activation='relu')(Fc)
        ans1 = Dense(units=channel/r, activation='linear')(ans1)
        return ans1

    out1 = MLP(out1, r, channel)
    out2 = MLP(out2, r, channel)
    out3 = out1+out2
    out3 = Dense(units=channel, activation='sigmoid')(out3)
    return out3  # this return the attention weight


# We define the spatial attention module
def spatial_attention_module(F,channels,ks):
    out1 = MaxPooling3D(pool_size=(1, 1, channels), strides=(1,1,1),data_format='channels_first')(F)
    out2 = AveragePooling3D(pool_size=(1, 1, channels),strides=(1,1,1),data_format='channels_first')(F)

    ans = Concatenate()([out1, out2])

    ans = Conv3D(filters=1, kernel_size=(7,7,7),
                 padding='same', activation='sigmoid')(ans)

    return ans


# We define our  CBAModule for our Neural Network
def CBAM(input, channels ,ks):
    McF = channel_attention_module(input, r, channels)
    Fprime = multiply([McF, input])
    print(Fprime.shape)
    MsF = spatial_attention_module((Fprime),channels,ks)
    Fprimeprime = multiply([MsF, Fprime])

    return Fprimeprime


# Now we define the neural network for AcsiNet


def NeuralNetwork(input):

    # Helper functions for my code
    def bluearrow(t):
        t = BatchNormalization()(t)
        t = LeakyReLU(alpha=0.1)(t)
        return t

    def greenarrow(t):
        t = BatchNormalization()(t)
        t = ReLU()(t)
        return t

    def pinkarrow(t, shaper):
        t = Reshape(shaper)(t)
        return t

    def ResidualBLock(input, kernel):
        output = bluearrow(input)
        side = output

        output = Conv3D(filters=24, kernel_size=kernel,
                        strides=(1, 1, 1), padding='same')(output)
        output = bluearrow(output)

        output = Conv3D(filters=48, kernel_size=kernel,
                        strides=(1, 1, 1), padding='same')(output)
        Attention = CBAM(output, 48,kernel)
        output = bluearrow(output)
        output = multiply([output, Attention])

        output = Conv3D(filters=6, kernel_size=kernel,
                        strides=(1, 1, 1), padding='same')(output)
        output = add([output, side])

        return output

    # Start the network
    output = Conv3D(filters=12, kernel_size=(2, 3, 2),
                    strides=(1, 1, 1), padding='same')(input)
    output = bluearrow(output)

    output = Conv3D(filters=24, kernel_size=(2, 3, 2),
                    strides=(1, 1, 1), padding='same')(output)
    Attention = CBAM(output, 24,(2,3,2))
    output = bluearrow(output)
    output = multiply([output, Attention])

    output = Conv3D(filters=4, kernel_size=(2, 3, 2),
                    strides=(1, 1, 1), padding='same')(output)
    Attention = CBAM(output, 4,(2,3,2))
    output = bluearrow(output)
    output = multiply([output, Attention])

    output = Conv3D(filters=1, kernel_size=(2, 3, 2),
                    strides=(1, 1, 1), padding='same')(output)
    output = bluearrow(output)
    output = Reshape((M*Nx*Ny,))(output)
    output = Dense(M*Nx*Ny*2, activation='linear')(output)
    output = greenarrow(output)
    output = pinkarrow(output, (Ny, M, Nx, 2))

    output = Conv3D(filters=6, kernel_size=(2, 3, 2),
                    strides=(1, 1, 1), padding='same')(output)

    # Implement the residual block
    output = ResidualBLock(output, (2, 3, 2))
    output = ResidualBLock(output, (3, 5, 3))

    Attention = CBAM(output, 6,(3,5,3))
    output = multiply([Attention, output])

    output = Conv3D(filters=2, kernel_size=(2, 3, 2),
                    strides=(1, 1, 1), padding='same')(output)

    return output



# from google.colab import drive
# drive.mount('/content/drive')

# Data loading for input
mat2 = mat73.loadmat('/content/drive/MyDrive/38.901_dual_train.mat')
x_train = mat2["HT"]  # array for accesing the HT variable in the matrix
mat1 = mat73.loadmat('/content/drive/MyDrive/38.901_dual_test.mat')
x_test = mat1["HT"]  # array for accesing the HT variable in the matrix



x_train = x_train.astype(float)
x_test = x_test.astype(float)


print(x_train.shape)
print(x_test.shape)

# (88889, 2, 26, 32, 2)
# (19111, 2, 26, 32, 2)


input_test_samples = np.array(x_test[:, :,:,:,0:1])
input_train_samples = np.array(x_train[:, :,:,:,0:1])
output_test_samples = np.array(x_test[:, :,:,:,1:2])
output_train_samples = np.array(x_train[:, :,:,:,1:2])


input_test_samples = np.reshape(input_test_samples,    (len(input_test_samples),   2,M,Nx,Ny))
output_test_samples = np.reshape(output_test_samples,  (len(output_test_samples),  2,M,Nx,Ny))
input_train_samples = np.reshape(input_train_samples,  (len(input_train_samples),  2,M,Nx,Ny))
output_train_samples= np.reshape(output_train_samples, (len(output_train_samples), 2,M,Nx,Ny))


print(input_test_samples.shape)
print(output_test_samples.shape)

print(input_train_samples.shape)
print(output_train_samples.shape)


# (19111, 2, 26, 8, 4)
# (19111, 2, 26, 8, 4)
# (88889, 2, 26, 8, 4)
# (88889, 2, 26, 8, 4)



output_train_samples=np.transpose(output_train_samples,(0,4,2,3,1))

input_train_samples=np.transpose(input_train_samples,(0,4,2,3,1))

output_test_samples=np.transpose(output_test_samples,(0,4,2,3,1))

input_test_samples=np.transpose(input_test_samples,(0,4,2,3,1))



print(input_test_samples.shape)
print(output_test_samples.shape)

print(input_train_samples.shape)
print(output_train_samples.shape)

# (19111, 4, 26, 8, 2)
# (19111, 4, 26, 8, 2)
# (88889, 4, 26, 8, 2)
# (88889, 4, 26, 8, 2)



input_tensor = Input(shape=(Ny, M, Nx, 2))
output_value = NeuralNetwork(input_tensor)

initial_lr = 1e-3
lr_reduction_factor = 0.1
lr_reduction_patience = 20
min_learning_rate = 1e-8

CsiNet = Model(inputs=[input_tensor], outputs=[output_value])
CsiNet.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=initial_lr), loss='mse')
print(CsiNet.summary())


# Define a learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=lr_reduction_factor,
    patience=lr_reduction_patience,
    min_lr=min_learning_rate,
    verbose=1
)

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


history = LossHistory()
# Create a file name
file = 'CsiNet_'+'_dim'+time.strftime('_%m_%d')
path = 'result/TensorBoard_%s' % file

#  We now train our model with our data with number of steps=1000, (samples per size =200) and shuffle data with each step
#  Also validate using x_val as both input and target
#  We also need to record the losses using callbacks
CsiNet.fit(input_train_samples, output_train_samples,
           epochs=50,
           batch_size=512,
           validation_data=(input_train_samples, output_train_samples),
           shuffle=True,
           callbacks=[history,
                      TensorBoard(log_dir=path), lr_scheduler])

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
    Hi_real = np.reshape(Hi[:, :, :, :, 0], (len(Hi), -1))
    Hi_img = np.reshape(Hi[:, :, :, :, 1], (len(Hi), -1))
    Hi_complex = (Hi_real-0.5)+1j*(Hi_img-0.5)

    Hi_hat_real = np.reshape(Hi_hat[:, :, :, :, 0], (len(Hi_hat), -1))
    Hi_hat_img = np.reshape(Hi_hat[:, :, :, :, 1], (len(Hi_hat), -1))
    Hi_hat_complex = (Hi_hat_real-0.5)+1j*(Hi_hat_img-0.5)

    mse = np.sum(np.square(np.abs(Hi_complex - Hi_hat_complex)), axis=(1))
    power = np.sum(np.abs(Hi_complex) ** 2, axis=(1))
    NMSE = 10 * np.log10(np.mean(mse) / np.maximum(np.mean(power), 1e-10))
    return NMSE



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



# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_1 (InputLayer)           [(None, 4, 26, 4, 1  0           []
#                                 )]

#  conv3d (Conv3D)                (None, 4, 26, 4, 12  156         ['input_1[0][0]']
#                                 )

#  batch_normalization (BatchNorm  (None, 4, 26, 4, 12  48         ['conv3d[0][0]']
#  alization)                     )

#  leaky_re_lu (LeakyReLU)        (None, 4, 26, 4, 12  0           ['batch_normalization[0][0]']
#                                 )

#  conv3d_1 (Conv3D)              (None, 4, 26, 4, 24  3480        ['leaky_re_lu[0][0]']
#                                 )

#  max_pooling3d (MaxPooling3D)   (None, 1, 1, 1, 24)  0           ['conv3d_1[0][0]']

#  average_pooling3d (AveragePool  (None, 1, 1, 1, 24)  0          ['conv3d_1[0][0]']
#  ing3D)

#  dense (Dense)                  (None, 1, 1, 1, 1)   25          ['max_pooling3d[0][0]']

#  dense_2 (Dense)                (None, 1, 1, 1, 1)   25          ['average_pooling3d[0][0]']

#  dense_1 (Dense)                (None, 1, 1, 1, 1)   2           ['dense[0][0]']

#  dense_3 (Dense)                (None, 1, 1, 1, 1)   2           ['dense_2[0][0]']

#  tf.__operators__.add (TFOpLamb  (None, 1, 1, 1, 1)  0           ['dense_1[0][0]',
#  da)                                                              'dense_3[0][0]']

#  dense_4 (Dense)                (None, 1, 1, 1, 24)  48          ['tf.__operators__.add[0][0]']

#  multiply (Multiply)            (None, 4, 26, 4, 24  0           ['dense_4[0][0]',
#                                 )                                 'conv3d_1[0][0]']

#  max_pooling3d_1 (MaxPooling3D)  (None, 4, 26, 4, 1)  0          ['multiply[0][0]']

#  average_pooling3d_1 (AveragePo  (None, 4, 26, 4, 1)  0          ['multiply[0][0]']
#  oling3D)

#  concatenate (Concatenate)      (None, 4, 26, 4, 2)  0           ['max_pooling3d_1[0][0]',
#                                                                   'average_pooling3d_1[0][0]']

#  batch_normalization_1 (BatchNo  (None, 4, 26, 4, 24  96         ['conv3d_1[0][0]']
#  rmalization)                   )

#  conv3d_2 (Conv3D)              (None, 4, 26, 4, 1)  687         ['concatenate[0][0]']

#  leaky_re_lu_1 (LeakyReLU)      (None, 4, 26, 4, 24  0           ['batch_normalization_1[0][0]']
#                                 )

#  multiply_1 (Multiply)          (None, 4, 26, 4, 24  0           ['conv3d_2[0][0]',
#                                 )                                 'multiply[0][0]']

#  multiply_2 (Multiply)          (None, 4, 26, 4, 24  0           ['leaky_re_lu_1[0][0]',
#                                 )                                 'multiply_1[0][0]']

#  conv3d_3 (Conv3D)              (None, 4, 26, 4, 4)  1156        ['multiply_2[0][0]']

#  max_pooling3d_2 (MaxPooling3D)  (None, 1, 1, 1, 4)  0           ['conv3d_3[0][0]']

#  average_pooling3d_2 (AveragePo  (None, 1, 1, 1, 4)  0           ['conv3d_3[0][0]']
#  oling3D)

#  dense_5 (Dense)                (None, 1, 1, 1, 0)   0           ['max_pooling3d_2[0][0]']

#  dense_7 (Dense)                (None, 1, 1, 1, 0)   0           ['average_pooling3d_2[0][0]']

#  dense_6 (Dense)                (None, 1, 1, 1, 0)   0           ['dense_5[0][0]']

#  dense_8 (Dense)                (None, 1, 1, 1, 0)   0           ['dense_7[0][0]']

#  tf.__operators__.add_1 (TFOpLa  (None, 1, 1, 1, 0)  0           ['dense_6[0][0]',
#  mbda)                                                            'dense_8[0][0]']

#  dense_9 (Dense)                (None, 1, 1, 1, 4)   4           ['tf.__operators__.add_1[0][0]']

#  multiply_3 (Multiply)          (None, 4, 26, 4, 4)  0           ['dense_9[0][0]',
#                                                                   'conv3d_3[0][0]']

#  max_pooling3d_3 (MaxPooling3D)  (None, 4, 26, 4, 1)  0          ['multiply_3[0][0]']

#  average_pooling3d_3 (AveragePo  (None, 4, 26, 4, 1)  0          ['multiply_3[0][0]']
#  oling3D)

#  concatenate_1 (Concatenate)    (None, 4, 26, 4, 2)  0           ['max_pooling3d_3[0][0]',
#                                                                   'average_pooling3d_3[0][0]']

#  batch_normalization_2 (BatchNo  (None, 4, 26, 4, 4)  16         ['conv3d_3[0][0]']
#  rmalization)

#  conv3d_4 (Conv3D)              (None, 4, 26, 4, 1)  687         ['concatenate_1[0][0]']

#  leaky_re_lu_2 (LeakyReLU)      (None, 4, 26, 4, 4)  0           ['batch_normalization_2[0][0]']

#  multiply_4 (Multiply)          (None, 4, 26, 4, 4)  0           ['conv3d_4[0][0]',
#                                                                   'multiply_3[0][0]']

#  multiply_5 (Multiply)          (None, 4, 26, 4, 4)  0           ['leaky_re_lu_2[0][0]',
#                                                                   'multiply_4[0][0]']

#  conv3d_5 (Conv3D)              (None, 4, 26, 4, 1)  49          ['multiply_5[0][0]']

#  batch_normalization_3 (BatchNo  (None, 4, 26, 4, 1)  4          ['conv3d_5[0][0]']
#  rmalization)

#  leaky_re_lu_3 (LeakyReLU)      (None, 4, 26, 4, 1)  0           ['batch_normalization_3[0][0]']

#  reshape (Reshape)              (None, 416)          0           ['leaky_re_lu_3[0][0]']

#  dense_10 (Dense)               (None, 832)          346944      ['reshape[0][0]']

#  batch_normalization_4 (BatchNo  (None, 832)         3328        ['dense_10[0][0]']
#  rmalization)

#  re_lu (ReLU)                   (None, 832)          0           ['batch_normalization_4[0][0]']

#  reshape_1 (Reshape)            (None, 4, 26, 4, 2)  0           ['re_lu[0][0]']

#  conv3d_6 (Conv3D)              (None, 4, 26, 4, 6)  150         ['reshape_1[0][0]']

#  batch_normalization_5 (BatchNo  (None, 4, 26, 4, 6)  24         ['conv3d_6[0][0]']
#  rmalization)

#  leaky_re_lu_4 (LeakyReLU)      (None, 4, 26, 4, 6)  0           ['batch_normalization_5[0][0]']

#  conv3d_7 (Conv3D)              (None, 4, 26, 4, 24  1752        ['leaky_re_lu_4[0][0]']
#                                 )

#  batch_normalization_6 (BatchNo  (None, 4, 26, 4, 24  96         ['conv3d_7[0][0]']
#  rmalization)                   )

#  leaky_re_lu_5 (LeakyReLU)      (None, 4, 26, 4, 24  0           ['batch_normalization_6[0][0]']
#                                 )

#  conv3d_8 (Conv3D)              (None, 4, 26, 4, 48  13872       ['leaky_re_lu_5[0][0]']
#                                 )

#  max_pooling3d_4 (MaxPooling3D)  (None, 1, 1, 1, 48)  0          ['conv3d_8[0][0]']

#  average_pooling3d_4 (AveragePo  (None, 1, 1, 1, 48)  0          ['conv3d_8[0][0]']
#  oling3D)

#  dense_11 (Dense)               (None, 1, 1, 1, 3)   147         ['max_pooling3d_4[0][0]']

#  dense_13 (Dense)               (None, 1, 1, 1, 3)   147         ['average_pooling3d_4[0][0]']

#  dense_12 (Dense)               (None, 1, 1, 1, 3)   12          ['dense_11[0][0]']

#  dense_14 (Dense)               (None, 1, 1, 1, 3)   12          ['dense_13[0][0]']

#  tf.__operators__.add_2 (TFOpLa  (None, 1, 1, 1, 3)  0           ['dense_12[0][0]',
#  mbda)                                                            'dense_14[0][0]']

#  dense_15 (Dense)               (None, 1, 1, 1, 48)  192         ['tf.__operators__.add_2[0][0]']

#  multiply_6 (Multiply)          (None, 4, 26, 4, 48  0           ['dense_15[0][0]',
#                                 )                                 'conv3d_8[0][0]']

#  max_pooling3d_5 (MaxPooling3D)  (None, 4, 26, 4, 1)  0          ['multiply_6[0][0]']

#  average_pooling3d_5 (AveragePo  (None, 4, 26, 4, 1)  0          ['multiply_6[0][0]']
#  oling3D)

#  concatenate_2 (Concatenate)    (None, 4, 26, 4, 2)  0           ['max_pooling3d_5[0][0]',
#                                                                   'average_pooling3d_5[0][0]']

#  batch_normalization_7 (BatchNo  (None, 4, 26, 4, 48  192        ['conv3d_8[0][0]']
#  rmalization)                   )

#  conv3d_9 (Conv3D)              (None, 4, 26, 4, 1)  687         ['concatenate_2[0][0]']

#  leaky_re_lu_6 (LeakyReLU)      (None, 4, 26, 4, 48  0           ['batch_normalization_7[0][0]']
#                                 )

#  multiply_7 (Multiply)          (None, 4, 26, 4, 48  0           ['conv3d_9[0][0]',
#                                 )                                 'multiply_6[0][0]']

#  multiply_8 (Multiply)          (None, 4, 26, 4, 48  0           ['leaky_re_lu_6[0][0]',
#                                 )                                 'multiply_7[0][0]']

#  conv3d_10 (Conv3D)             (None, 4, 26, 4, 6)  3462        ['multiply_8[0][0]']

#  add (Add)                      (None, 4, 26, 4, 6)  0           ['conv3d_10[0][0]',
#                                                                   'leaky_re_lu_4[0][0]']

#  batch_normalization_8 (BatchNo  (None, 4, 26, 4, 6)  24         ['add[0][0]']
#  rmalization)

#  leaky_re_lu_7 (LeakyReLU)      (None, 4, 26, 4, 6)  0           ['batch_normalization_8[0][0]']

#  conv3d_11 (Conv3D)             (None, 4, 26, 4, 24  6504        ['leaky_re_lu_7[0][0]']
#                                 )

#  batch_normalization_9 (BatchNo  (None, 4, 26, 4, 24  96         ['conv3d_11[0][0]']
#  rmalization)                   )

#  leaky_re_lu_8 (LeakyReLU)      (None, 4, 26, 4, 24  0           ['batch_normalization_9[0][0]']
#                                 )

#  conv3d_12 (Conv3D)             (None, 4, 26, 4, 48  51888       ['leaky_re_lu_8[0][0]']
#                                 )

#  max_pooling3d_6 (MaxPooling3D)  (None, 1, 1, 1, 48)  0          ['conv3d_12[0][0]']

#  average_pooling3d_6 (AveragePo  (None, 1, 1, 1, 48)  0          ['conv3d_12[0][0]']
#  oling3D)

#  dense_16 (Dense)               (None, 1, 1, 1, 3)   147         ['max_pooling3d_6[0][0]']

#  dense_18 (Dense)               (None, 1, 1, 1, 3)   147         ['average_pooling3d_6[0][0]']

#  dense_17 (Dense)               (None, 1, 1, 1, 3)   12          ['dense_16[0][0]']

#  dense_19 (Dense)               (None, 1, 1, 1, 3)   12          ['dense_18[0][0]']

#  tf.__operators__.add_3 (TFOpLa  (None, 1, 1, 1, 3)  0           ['dense_17[0][0]',
#  mbda)                                                            'dense_19[0][0]']

#  dense_20 (Dense)               (None, 1, 1, 1, 48)  192         ['tf.__operators__.add_3[0][0]']

#  multiply_9 (Multiply)          (None, 4, 26, 4, 48  0           ['dense_20[0][0]',
#                                 )                                 'conv3d_12[0][0]']

#  max_pooling3d_7 (MaxPooling3D)  (None, 4, 26, 4, 1)  0          ['multiply_9[0][0]']

#  average_pooling3d_7 (AveragePo  (None, 4, 26, 4, 1)  0          ['multiply_9[0][0]']
#  oling3D)

#  concatenate_3 (Concatenate)    (None, 4, 26, 4, 2)  0           ['max_pooling3d_7[0][0]',
#                                                                   'average_pooling3d_7[0][0]']

#  batch_normalization_10 (BatchN  (None, 4, 26, 4, 48  192        ['conv3d_12[0][0]']
#  ormalization)                  )

#  conv3d_13 (Conv3D)             (None, 4, 26, 4, 1)  687         ['concatenate_3[0][0]']

#  leaky_re_lu_9 (LeakyReLU)      (None, 4, 26, 4, 48  0           ['batch_normalization_10[0][0]']
#                                 )

#  multiply_10 (Multiply)         (None, 4, 26, 4, 48  0           ['conv3d_13[0][0]',
#                                 )                                 'multiply_9[0][0]']

#  multiply_11 (Multiply)         (None, 4, 26, 4, 48  0           ['leaky_re_lu_9[0][0]',
#                                 )                                 'multiply_10[0][0]']

#  conv3d_14 (Conv3D)             (None, 4, 26, 4, 6)  12966       ['multiply_11[0][0]']

#  add_1 (Add)                    (None, 4, 26, 4, 6)  0           ['conv3d_14[0][0]',
#                                                                   'leaky_re_lu_7[0][0]']

#  max_pooling3d_8 (MaxPooling3D)  (None, 1, 1, 1, 6)  0           ['add_1[0][0]']

#  average_pooling3d_8 (AveragePo  (None, 1, 1, 1, 6)  0           ['add_1[0][0]']
#  oling3D)

#  dense_21 (Dense)               (None, 1, 1, 1, 0)   0           ['max_pooling3d_8[0][0]']

#  dense_23 (Dense)               (None, 1, 1, 1, 0)   0           ['average_pooling3d_8[0][0]']

#  dense_22 (Dense)               (None, 1, 1, 1, 0)   0           ['dense_21[0][0]']

#  dense_24 (Dense)               (None, 1, 1, 1, 0)   0           ['dense_23[0][0]']

#  tf.__operators__.add_4 (TFOpLa  (None, 1, 1, 1, 0)  0           ['dense_22[0][0]',
#  mbda)                                                            'dense_24[0][0]']

#  dense_25 (Dense)               (None, 1, 1, 1, 6)   6           ['tf.__operators__.add_4[0][0]']

#  multiply_12 (Multiply)         (None, 4, 26, 4, 6)  0           ['dense_25[0][0]',
#                                                                   'add_1[0][0]']

#  max_pooling3d_9 (MaxPooling3D)  (None, 4, 26, 4, 1)  0          ['multiply_12[0][0]']

#  average_pooling3d_9 (AveragePo  (None, 4, 26, 4, 1)  0          ['multiply_12[0][0]']
#  oling3D)

#  concatenate_4 (Concatenate)    (None, 4, 26, 4, 2)  0           ['max_pooling3d_9[0][0]',
#                                                                   'average_pooling3d_9[0][0]']

#  conv3d_15 (Conv3D)             (None, 4, 26, 4, 1)  687         ['concatenate_4[0][0]']

#  multiply_13 (Multiply)         (None, 4, 26, 4, 6)  0           ['conv3d_15[0][0]',
#                                                                   'multiply_12[0][0]']

#  multiply_14 (Multiply)         (None, 4, 26, 4, 6)  0           ['multiply_13[0][0]',
#                                                                   'add_1[0][0]']

#  conv3d_16 (Conv3D)             (None, 4, 26, 4, 2)  146         ['multiply_14[0][0]']

# ==================================================================================================
# Total params: 451,208
# Trainable params: 449,150
# Non-trainable params: 2,058
# __________________________________________________________________________________________________
# None

# Epoch 1/50
#   6/174 [>.............................] - ETA: 16s - loss: 9363.9932 WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0428s vs `on_train_batch_end` time: 0.0911s). Check your callbacks.
# 174/174 [==============================] - 58s 161ms/step - loss: 323.6438 - val_loss: 2.1468e-04 - lr: 0.0010
# Epoch 2/50
# 174/174 [==============================] - 21s 121ms/step - loss: 0.0045 - val_loss: 2.6625e-04 - lr: 0.0010
# Epoch 3/50
# 174/174 [==============================] - 21s 120ms/step - loss: 0.0021 - val_loss: 2.9248e-04 - lr: 0.0010
# Epoch 4/50
# 174/174 [==============================] - 21s 120ms/step - loss: 0.0013 - val_loss: 0.0011 - lr: 0.0010
# Epoch 5/50
# 174/174 [==============================] - 21s 121ms/step - loss: 8.9199e-04 - val_loss: 0.0019 - lr: 0.0010
# Epoch 6/50
# 174/174 [==============================] - 21s 121ms/step - loss: 6.9151e-04 - val_loss: 0.0018 - lr: 0.0010
# Epoch 7/50
# 174/174 [==============================] - 21s 121ms/step - loss: 5.6962e-04 - val_loss: 0.0020 - lr: 0.0010
# Epoch 8/50
# 174/174 [==============================] - 21s 120ms/step - loss: 4.8112e-04 - val_loss: 0.0019 - lr: 0.0010
# Epoch 9/50
# 174/174 [==============================] - 21s 121ms/step - loss: 4.1670e-04 - val_loss: 0.0017 - lr: 0.0010
# Epoch 10/50
# 174/174 [==============================] - 21s 121ms/step - loss: 3.6677e-04 - val_loss: 0.0033 - lr: 0.0010
# Epoch 11/50
# 174/174 [==============================] - 21s 120ms/step - loss: 3.2652e-04 - val_loss: 0.0220 - lr: 0.0010
# Epoch 12/50
# 174/174 [==============================] - 21s 121ms/step - loss: 2.9619e-04 - val_loss: 0.0047 - lr: 0.0010
# Epoch 13/50
# 174/174 [==============================] - 21s 121ms/step - loss: 2.7244e-04 - val_loss: 2.3420 - lr: 0.0010
# Epoch 14/50
# 174/174 [==============================] - 21s 121ms/step - loss: 2.5282e-04 - val_loss: 0.0028 - lr: 0.0010
# Epoch 15/50
# 174/174 [==============================] - 21s 120ms/step - loss: 2.5064e-04 - val_loss: 0.0020 - lr: 0.0010
# Epoch 16/50
# 174/174 [==============================] - 21s 120ms/step - loss: 2.1353e-04 - val_loss: 0.0015 - lr: 0.0010
# Epoch 17/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.9989e-04 - val_loss: 0.0023 - lr: 0.0010
# Epoch 18/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.8046e-04 - val_loss: 0.0017 - lr: 0.0010
# Epoch 19/50
# 174/174 [==============================] - 21s 121ms/step - loss: 1.6874e-04 - val_loss: 9.4556e-04 - lr: 0.0010
# Epoch 20/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.5791e-04 - val_loss: 0.0011 - lr: 0.0010
# Epoch 21/50
# 174/174 [==============================] - ETA: 0s - loss: 1.4783e-04
# Epoch 21: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
# 174/174 [==============================] - 21s 120ms/step - loss: 1.4783e-04 - val_loss: 0.0011 - lr: 0.0010
# Epoch 22/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.4250e-04 - val_loss: 3.3674e-04 - lr: 1.0000e-04
# Epoch 23/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.4127e-04 - val_loss: 1.5401e-04 - lr: 1.0000e-04
# Epoch 24/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.3991e-04 - val_loss: 1.5794e-04 - lr: 1.0000e-04
# Epoch 25/50
# 174/174 [==============================] - 21s 119ms/step - loss: 1.3914e-04 - val_loss: 1.6605e-04 - lr: 1.0000e-04
# Epoch 26/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.3810e-04 - val_loss: 1.8554e-04 - lr: 1.0000e-04
# Epoch 27/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.3562e-04 - val_loss: 1.5062e-04 - lr: 1.0000e-04
# Epoch 28/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.3561e-04 - val_loss: 1.4978e-04 - lr: 1.0000e-04
# Epoch 29/50
# 174/174 [==============================] - 21s 119ms/step - loss: 1.3434e-04 - val_loss: 1.5772e-04 - lr: 1.0000e-04
# Epoch 30/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.3334e-04 - val_loss: 1.6795e-04 - lr: 1.0000e-04
# Epoch 31/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.3184e-04 - val_loss: 1.8265e-04 - lr: 1.0000e-04
# Epoch 32/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.3041e-04 - val_loss: 1.6294e-04 - lr: 1.0000e-04
# Epoch 33/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.2869e-04 - val_loss: 2.0032e-04 - lr: 1.0000e-04
# Epoch 34/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.2689e-04 - val_loss: 2.8008e-04 - lr: 1.0000e-04
# Epoch 35/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.2557e-04 - val_loss: 1.5074e-04 - lr: 1.0000e-04
# Epoch 36/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.2378e-04 - val_loss: 1.4734e-04 - lr: 1.0000e-04
# Epoch 37/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.2211e-04 - val_loss: 1.7871e-04 - lr: 1.0000e-04
# Epoch 38/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.2027e-04 - val_loss: 1.4906e-04 - lr: 1.0000e-04
# Epoch 39/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.1848e-04 - val_loss: 1.4816e-04 - lr: 1.0000e-04
# Epoch 40/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.1618e-04 - val_loss: 1.4984e-04 - lr: 1.0000e-04
# Epoch 41/50
# 174/174 [==============================] - ETA: 0s - loss: 1.1405e-04
# Epoch 41: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
# 174/174 [==============================] - 21s 120ms/step - loss: 1.1405e-04 - val_loss: 1.4315e-04 - lr: 1.0000e-04
# Epoch 42/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.1270e-04 - val_loss: 1.1567e-04 - lr: 1.0000e-05
# Epoch 43/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.1255e-04 - val_loss: 1.1497e-04 - lr: 1.0000e-05
# Epoch 44/50
# 174/174 [==============================] - 21s 119ms/step - loss: 1.1221e-04 - val_loss: 1.1455e-04 - lr: 1.0000e-05
# Epoch 45/50
# 174/174 [==============================] - 21s 119ms/step - loss: 1.1199e-04 - val_loss: 1.1424e-04 - lr: 1.0000e-05
# Epoch 46/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.1177e-04 - val_loss: 1.1422e-04 - lr: 1.0000e-05
# Epoch 47/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.1105e-04 - val_loss: 1.1349e-04 - lr: 1.0000e-05
# Epoch 48/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.1131e-04 - val_loss: 1.1324e-04 - lr: 1.0000e-05
# Epoch 49/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.1087e-04 - val_loss: 1.1302e-04 - lr: 1.0000e-05
# Epoch 50/50
# 174/174 [==============================] - 21s 120ms/step - loss: 1.1067e-04 - val_loss: 1.1269e-04 - lr: 1.0000e-05
# 598/598 [==============================] - 7s 10ms/step
# It cost 0.000404 sec
# NMSE is  -17.615643820318258
