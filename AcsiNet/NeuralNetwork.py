import tensorflow as tf
from keras.layers import Conv3D, add, BatchNormalization, LeakyReLU, ReLU, Flatten, Reshape, GlobalMaxPooling3D, MaxPooling3D, multiply, Concatenate, GlobalAveragePooling3D, AveragePooling3D, AveragePooling3D, Dense
from keras.models import Model
from math import floor
import numpy as np
# Now we store the dimensions of the input.

M = 26
Nx = 8
Ny = 4
r = 16


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

