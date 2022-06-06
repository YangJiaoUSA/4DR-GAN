from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# import tensorflow_addons as tfa
# import BatchInstanceNorm


import os
import time
OUTPUT_CHANNELS = 1

def downsample(inputs, filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  x = tf.keras.layers.Conv2D(filters, size, strides=[2, 2], padding='same',
                             kernel_initializer=initializer, use_bias=False)(inputs)
  res = tf.keras.layers.Conv2D(filters, [1, 1], strides=[2, 2], padding='same',
                             kernel_initializer=initializer, use_bias=False)(inputs)
  x = tf.keras.layers.LeakyReLU()(tf.keras.layers.BatchNormalization()(x))
  x = tf.keras.layers.add([x, res])
  x = tf.keras.layers.LeakyReLU()(x)

  return x
def upsample(inputs, filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  x = tf.keras.layers.Conv2DTranspose(filters, size, strides=[2, 2],
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  # res = tf.keras.layers.Conv2D(filters, [1, 1], strides=[2, 2], padding='same')(inputs)
  # x = tf.keras.layers.add([x, res])

  x = tf.keras.layers.Dropout(0.5)(x)

  x = tf.keras.layers.ReLU()(x)

  return x

def batch_instance_norm(x, scope='batch_instance_norm'):
    with tf.compat.v1.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2, 3], keepdims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ins = (x - ins_mean) / (tf.math.sqrt(ins_sigma + eps))

        rho = tf.compat.v1.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0), constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.compat.v1.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.compat.v1.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_batch + (1 - rho) * x_ins
        x_hat = x_hat * gamma + beta

        return x_hat
def normalization_layer(x):
    out=tf.keras.layers.BatchNormalization()(x)
    return out


def Generator_encoder_video_frame():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    initializer = tf.random_normal_initializer(0., 0.02)

    def downconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(x))
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        # x2 = tf.keras.layers.Conv3D(128, size, strides=[1, 1], padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(inputs2)
        # x2 = tf.keras.layers.ReLU()(x2)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(inputs2)
        x = tf.math.multiply(inputs1, x2_m)
        x = tf.math.add(x, x2_a)
        return x


    def spatial_encoder(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 64, 4, [2, 2, 1])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 1])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)

        skips = reversed(skips[:-1])

        return x, skips

    conved_output1, skips1 = spatial_encoder(inputs1)
    conved_output2, skips2 = spatial_encoder(inputs2)

    x = conved_output1

    channel_list = [512, 512, 512, 512, 256, 128, 64]
    size_list = [4, 4, 4, 4, 4, 4, 4]
    stride_list = [[2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2]]
    skip_list = [True, True, True, True, True, True, True]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for channel, size, stride, drop, skip_d, skip1, skip2, norm in zip(channel_list, size_list, stride_list, drop_list, skip_list, skips1, skips2,
                                                        norm_list):
        x = tf.keras.layers.Conv3DTranspose(channel, size, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)
        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            x = tf.keras.layers.BatchNormalization()(x)
            res = tf.keras.layers.Conv3D(channel, 4, strides=1, padding='same',
                                       kernel_initializer=initializer, use_bias=False)(x)
            res = tf.keras.layers.ReLU()(res)
            x = spade_norm(x, skip1, channel, 4)
            x = spade_norm(x, skip2, channel, 4)
            x = tf.keras.layers.Conv3D(channel, 4, strides=1, padding='same',
                                       kernel_initializer=initializer, use_bias=False)(x)
            x = tf.keras.layers.ReLU()(x)

            # x1 = spade_norm(x, skip1, channel, 4)
            # x2 = spade_norm(x1, skip2, channel, 4)
            x = tf.keras.layers.Add()([x, res])

        x = tf.keras.layers.ReLU()(x)
        # if skip_d is True:
            # x = tf.keras.layers.Concatenate()([x, skip1])

    x = tf.keras.layers.Conv3DTranspose(1, 4, strides=[2,2,1],
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=[inputs1, inputs2], outputs=x)

def Generator_noafs_video_frame():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    initializer = tf.random_normal_initializer(0., 0.02)

    def downconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(x))
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        # x2 = tf.keras.layers.Conv3D(128, size, strides=[1, 1], padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(inputs2)
        # x2 = tf.keras.layers.ReLU()(x2)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(inputs2)
        x = tf.math.multiply(inputs1, x2_m)
        x = tf.math.add(x, x2_a)
        return x


    def spatial_encoder(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 64, 4, [2, 2, 1])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 1])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)

        skips = reversed(skips[:-1])

        return x, skips

    conved_output1, skips1 = spatial_encoder(inputs1)
    conved_output2, skips2 = spatial_encoder(inputs2)

    x = conved_output1

    channel_list = [512, 512, 512, 512, 256, 128, 64]
    size_list = [4, 4, 4, 4, 4, 4, 4]
    stride_list = [[2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2]]
    skip_list = [True, True, True, True, True, True, True]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for channel, size, stride, drop, skip_d, skip1, skip2, norm in zip(channel_list, size_list, stride_list, drop_list, skip_list, skips1, skips2,
                                                        norm_list):
        x = tf.keras.layers.Conv3DTranspose(channel, size, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)
        # x = tf.keras.layers.ReLU()(normalization_layer(x))
        # x = tf.keras.layers.Conv2D(size, 3, strides=1, padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(x)
        # x = normalization_layer(x)
        # res = tf.keras.layers.Conv3D(size, [1, 1, 1], strides=[2, 2, 1], padding='same')(x)
        # x = tf.keras.layers.add([x1, res])
        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            x = tf.keras.layers.BatchNormalization()(x)
            # x = spade_norm(x, skip1, channel, 4)
            x = spade_norm(x, skip2, channel, 4)
            # x = spade_norm_mi(x, inputs3, channel, 4)
            # x = BatchInstanceNorm.BatchInstanceNormalization()(x)

        x = tf.keras.layers.ReLU()(x)
        if skip_d is True:
            # skip1 = tf.keras.layers.Conv2D(channel, size, strides=[1, 1], padding='same',
            #                               kernel_initializer=initializer, use_bias=False)(skip1)
            x = tf.keras.layers.Concatenate()([x, skip1])

    x = tf.keras.layers.Conv3DTranspose(1, 4, strides=[2,2,1],
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=[inputs1, inputs2], outputs=x)

def Generator_encoder_video_frame_trans():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    initializer = tf.random_normal_initializer(0., 0.02)

    def downconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(x))
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def upconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3DTranspose(channels, sizes, strides=strides,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(inputs)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        # x2 = tf.keras.layers.Conv3D(128, size, strides=[1, 1], padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(inputs2)
        # x2 = tf.keras.layers.ReLU()(x2)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(inputs2)
        x = tf.math.multiply(inputs1, x2_m)
        x = tf.math.add(x, x2_a)
        return x


    def spatial_encoder1(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 64, 4, [2, 2, 1])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 1])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)
        skips = reversed(skips[:-1])
        return x, skips

    def spatial_encoder2(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 32, 4, [2, 2, 1])  # [128, 128, 16]
        # skips.append(x)
        x = downconv(x, 64, 4, [2, 2, 2])  # [64, 64, 8]
        # skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 1])  # [32, 32, 8]
        # skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 2])  # [16, 16, 4]
        # skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 1])  # [8, 8, 4]
        # skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 2])  # [4, 4, 2]
        # skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 1])  # [2, 2, 2]
        # skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 2])  # [1, 1, 1]
        # skips.append(x)
        x = upconv(x, 256, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)
        x = upconv(x, 256, 4, [2, 2, 1])  # [1, 1, 1]
        skips.append(x)
        x = upconv(x, 256, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)
        x = upconv(x, 256, 4, [2, 2, 1])  # [1, 1, 1]
        skips.append(x)
        x = upconv(x, 128, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)
        x = upconv(x, 64, 4, [2, 2, 1])  # [1, 1, 1]
        skips.append(x)
        x = upconv(x, 32, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)
        # skips = reversed(skips[:-1])
        return x, skips

    conved_output1, skips1 = spatial_encoder1(inputs1)
    conved_output2, skips2 = spatial_encoder2(inputs2)

    x = conved_output1

    channel_list = [512, 512, 512, 512, 256, 128, 64]
    size_list = [4, 4, 4, 4, 4, 4, 4]
    stride_list = [[2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2]]
    skip_list = [True, True, True, True, True, True, True]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for channel, size, stride, drop, skip_d, skip1, skip2, norm in zip(channel_list, size_list, stride_list, drop_list, skip_list, skips1, skips2,
                                                        norm_list):
        x = tf.keras.layers.Conv3DTranspose(channel, size, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)
        # x = tf.keras.layers.ReLU()(normalization_layer(x))
        # x = tf.keras.layers.Conv2D(size, 3, strides=1, padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(x)
        # x = normalization_layer(x)
        # res = tf.keras.layers.Conv3D(size, [1, 1, 1], strides=[2, 2, 1], padding='same')(x)
        # x = tf.keras.layers.add([x1, res])
        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            x = tf.keras.layers.BatchNormalization()(x)
            x = spade_norm(x, skip1, channel, 4)
            x = spade_norm(x, skip2, channel, 4)
            # x = spade_norm_mi(x, inputs3, channel, 4)
            # x = BatchInstanceNorm.BatchInstanceNormalization()(x)

        x = tf.keras.layers.ReLU()(x)
        # if skip_d is True:
            # skip1 = tf.keras.layers.Conv2D(channel, size, strides=[1, 1], padding='same',
            #                               kernel_initializer=initializer, use_bias=False)(skip1)
            # x = tf.keras.layers.Concatenate()([x, skip1])

    x = tf.keras.layers.Conv3DTranspose(1, 4, strides=[2,2,1],
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=[inputs1, inputs2], outputs=x)

def Generator_encoder_video_frame3():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs3 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    initializer = tf.random_normal_initializer(0., 0.02)

    def downconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(x))
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        # x2 = tf.keras.layers.Conv3D(128, size, strides=[1, 1], padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(inputs2)
        # x2 = tf.keras.layers.ReLU()(x2)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(inputs2)
        x = tf.math.multiply(inputs1, x2_m)
        x = tf.math.add(x, x2_a)
        return x


    def spatial_encoder(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 64, 4, [2, 2, 1])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 1])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)

        skips = reversed(skips[:-1])

        return x, skips

    def spatial_encoder_s(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 32, 4, [2, 2, 1])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 64, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 1])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 2])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)

        skips = reversed(skips[:-1])

        return x, skips

    conved_output1, skips1 = spatial_encoder(inputs1)
    conved_output2, skips2 = spatial_encoder_s(inputs2)
    conved_output3, skips3 = spatial_encoder_s(inputs3)

    x = conved_output1

    channel_list = [512, 512, 512, 512, 256, 128, 64]
    size_list = [4, 4, 4, 4, 4, 4, 4]
    stride_list = [[2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2]]
    skip_list = [True, True, True, True, True, True, True]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for channel, size, stride, drop, skip_d, skip1, skip2, skip3, norm in zip(channel_list, size_list, stride_list, drop_list, skip_list, skips1, skips2, skips3,
                                                        norm_list):
        x = tf.keras.layers.Conv3DTranspose(channel, size, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)
        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            x = tf.keras.layers.BatchNormalization()(x)
            x = spade_norm(x, skip1, channel, 4)
            x = spade_norm(x, skip2, channel, 4)
            x = spade_norm(x, skip3, channel, 4)

        x = tf.keras.layers.ReLU()(x)
        # if skip_d is True:
            # x = tf.keras.layers.Concatenate()([x, skip1])

    x = tf.keras.layers.Conv3DTranspose(1, 4, strides=[2,2,1],
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=x)

def Generator_encoder_video_frame3_trans():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs3 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    initializer = tf.random_normal_initializer(0., 0.02)

    def downconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def downconv_spade(inputs, skip1, skip2, channels, sizes, strides):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = spade_norm(x, skip1, channels, 4)
        # x = spade_norm(x, skip2, channels, 4)
        x = tf.keras.layers.ReLU()(x)
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def upconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3DTranspose(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(x))
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def upconv_spade(input, skip1, skip2, skip3, channels, sizes, strides):
        x = tf.keras.layers.Conv3DTranspose(channels, sizes, strides=strides,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(input)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = spade_norm(x, skip1, channels, 4)
        x = spade_norm(x, skip2, channels, 4)
        # x = spade_norm(x, skip3, channels, 4)

        x = tf.keras.layers.ReLU()(x)
        return x

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        # x2 = tf.keras.layers.Conv3D(128, size, strides=[1, 1], padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(inputs2)
        # x2 = tf.keras.layers.ReLU()(x2)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(inputs2)
        x = tf.math.multiply(inputs1, x2_m)
        x = tf.math.add(x, x2_a)
        return x


    def spatial_encoder(inputs1, skips2, skips3):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv_spade(x, skips2[0], skips3[0], 64, 4, [2, 2, 1])  # [128, 128, 16]
        skips.append(x)
        x = downconv_spade(x, skips2[1], skips3[1], 128, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv_spade(x, skips2[2], skips3[2], 256, 4, [2, 2, 1])  # [32, 32, 8]
        skips.append(x)
        x = downconv_spade(x, skips2[3], skips3[3], 512, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv_spade(x, skips2[4], skips3[4], 512, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv_spade(x, skips2[5], skips3[5], 512, 4, [2, 2, 2])  # [4, 4, 2]
        skips.append(x)
        x = downconv_spade(x, skips2[6], skips3[6], 512, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv_spade(x, skips2[7], skips3[7], 512, 4, [2, 2, 2])  # [1, 1, 1]


        x = upconv_spade(x, skips[6], skips2[8], skips3[8], 512, 4, [2, 2, 2])
        x = upconv_spade(x, skips[5], skips2[9], skips3[9], 512, 4, [2, 2, 1])
        x = upconv_spade(x, skips[4], skips2[10], skips3[10], 512, 4, [2, 2, 2])
        x = upconv_spade(x, skips[3], skips2[11], skips3[11], 512, 4, [2, 2, 1])
        x = upconv_spade(x, skips[2], skips2[12], skips3[12], 256, 4, [2, 2, 2])
        x = upconv_spade(x, skips[1], skips2[13], skips3[13], 128, 4, [2, 2, 1])
        x = upconv_spade(x, skips[0], skips2[14], skips3[14], 64, 4, [2, 2, 2])

        return x

    def spatial_encoder_trans(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 16, 4, [2, 2, 1])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 32, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 64, 4, [2, 2, 1])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)
        x = upconv(x, 128, 4, [2, 2, 2])  # [2, 2, 2]
        skips.append(x)
        x = upconv(x, 128, 4, [2, 2,1])  # [4, 1, 1]
        skips.append(x)
        x = upconv(x, 128, 4, [2, 2, 2])  # [8, 1, 1]
        skips.append(x)
        x = upconv(x, 128, 4, [2, 2, 1])  # [16, 1, 1]
        skips.append(x)
        x = upconv(x, 64, 4, [2, 2, 2])  # [32, 1, 1]
        skips.append(x)
        x = upconv(x, 32, 4, [2, 2, 1])  # [64, 1, 1]
        skips.append(x)
        x = upconv(x, 16, 4, [2, 2, 2])  # [128, 1, 1]
        skips.append(x)

        # skips = reversed(skips[:-1])

        return x, skips


    conved_output2, skips2 = spatial_encoder_trans(inputs2)
    # conved_output3, skips3 = spatial_encoder_trans(inputs3)
    conved_output1 = spatial_encoder(inputs1, skips2, skips2)

    x = conved_output1

    x = tf.keras.layers.Conv3DTranspose(1, 4, strides=[2,2,1],
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=x)

def Generator_encoder_video_frame4():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs3 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs4 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    initializer = tf.random_normal_initializer(0., 0.02)

    def downconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(x))
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        # x2 = tf.keras.layers.Conv3D(128, size, strides=[1, 1], padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(inputs2)
        # x2 = tf.keras.layers.ReLU()(x2)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(inputs2)
        x = tf.math.multiply(inputs1, x2_m)
        x = tf.math.add(x, x2_a)
        return x


    def spatial_encoder(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 64, 4, [2, 2, 1])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 1])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)

        skips = reversed(skips[:-1])

        return x, skips

    def spatial_encoder_s(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 16, 4, [2, 2, 1])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 32, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 64, 4, [2, 2, 1])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)

        skips = reversed(skips[:-1])

        return x, skips

    conved_output1, skips1 = spatial_encoder(inputs1)
    conved_output2, skips2 = spatial_encoder_s(inputs2)
    conved_output3, skips3 = spatial_encoder_s(inputs3)
    conved_output4, skips4 = spatial_encoder_s(inputs4)

    x = conved_output1

    channel_list = [512, 512, 512, 512, 256, 128, 64]
    size_list = [4, 4, 4, 4, 4, 4, 4]
    stride_list = [[2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2]]
    skip_list = [True, True, True, True, True, True, True]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for channel, size, stride, drop, skip_d, skip1, skip2, skip3, skip4, norm in zip(channel_list, size_list, stride_list, drop_list, skip_list, skips1, skips2, skips3, skips4,
                                                        norm_list):
        x = tf.keras.layers.Conv3DTranspose(channel, size, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)
        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            x = tf.keras.layers.BatchNormalization()(x)
            x1 = spade_norm(x, skip1, channel, 4)
            x2 = spade_norm(x, skip2, channel, 4)
            x3 = spade_norm(x, skip3, channel, 4)
            x4 = spade_norm(x, skip4, channel, 4)
            x = tf.keras.layers.Concatenate(axis=-1)([x1, x2, x3, x4])

        x = tf.keras.layers.ReLU()(x)
        # if skip_d is True:
            # x = tf.keras.layers.Concatenate()([x, skip1])

    x = tf.keras.layers.Conv3DTranspose(1, 4, strides=[2,2,1],
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=x)


def Generator_lstm_video_frame2():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs3 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs4 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    initializer = tf.random_normal_initializer(0., 0.02)

    def downconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(x))
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def upconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3DTranspose(channels, sizes, strides=strides,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(inputs)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = tf.keras.layers.Conv3D(128, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2 = tf.keras.layers.ReLU()(x2)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(x2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(x2)
        x = tf.math.multiply(inputs1, x2_m)
        x = tf.math.add(x, x2_a)
        # x = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(x)
        # x = tf.keras.layers.ReLU()(x)
        return x


    def spatial_encoder1(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 64, 4, [2, 2, 1])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 1])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [1, 1, 1]
        skips.append(x)
        skips = reversed(skips[:-1])
        return x, skips

    with tf.compat.v1.variable_scope('scope1', reuse=True):
        conved_output1_1, skips1_1 = spatial_encoder1(inputs1)
        conved_output1_2, skips1_2 = spatial_encoder1(inputs2)
        conved_output1_3, skips1_3 = spatial_encoder1(inputs3)


    lstm_inputs = tf.keras.layers.Concatenate(axis=1)([conved_output1_1, conved_output1_2, conved_output1_3])

    encoder = tf.keras.layers.ConvLSTM2D(filters=512, kernel_size=4, strides=1, padding='same',
                                      data_format=None, dilation_rate=(1, 1), activation='relu',
                                      return_sequences=True, stateful=False)
    decoder = tf.keras.layers.ConvLSTM2D(filters=512, kernel_size=4, strides=1, padding='same',
                                      data_format=None, dilation_rate=(1, 1), activation='relu',
                                      return_sequences=False)
    # encoder.reset_states(states=None)
    encoded_state = encoder(inputs=lstm_inputs)
    decoded_state = decoder(inputs=encoded_state)
    x = decoded_state

    x = tf.expand_dims(x, axis=1)

    with tf.compat.v1.variable_scope('scope2', reuse=True):
        conved_output4, skips4 = spatial_encoder1(inputs4)


    channel_list = [512, 512, 512, 512, 256, 128, 64]
    size_list = [4, 4, 4, 4, 4, 4, 4]
    stride_list = [[2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2], [2,2,1], [2,2,2]]
    skip_list = [True, True, True, True, True, True, True]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for channel, size, stride, drop, skip_d, skip1, skip2, norm in zip(channel_list, size_list, stride_list, drop_list, skip_list, skips1_3, skips4,
                                                        norm_list):
        x = tf.keras.layers.Conv3DTranspose(channel, size, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)
        # x = tf.keras.layers.ReLU()(normalization_layer(x))
        # x = tf.keras.layers.Conv2D(size, 3, strides=1, padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(x)
        # x = normalization_layer(x)
        # res = tf.keras.layers.Conv3D(size, [1, 1, 1], strides=[2, 2, 1], padding='same')(x)
        # x = tf.keras.layers.add([x1, res])
        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            x = tf.keras.layers.BatchNormalization()(x)
            # res = tf.keras.layers.Conv3D(channel, 4, strides=1, padding='same',
            #                              kernel_initializer=initializer, use_bias=False)(x)
            # res = tf.keras.layers.ReLU()(res)
            # x = spade_norm(x, skip1, channel, 4)
            # x = spade_norm(x, skip2, channel, 4)
            # x = tf.keras.layers.Conv3D(channel, 4, strides=1, padding='same',
            #                            kernel_initializer=initializer, use_bias=False)(x)
            # x = tf.keras.layers.ReLU()(x)
            #
            # x = tf.keras.layers.Add()([x, res])

            res = x
            x = spade_norm(x, skip1, channel, 4)
            # x = tf.keras.layers.Add()([x, res])
            # res = x
            x = spade_norm(x, skip2, channel, 4)
            x = tf.keras.layers.Add()([x, res])

        x = tf.keras.layers.ReLU()(x)
        # if skip_d is True:
            # skip1 = tf.keras.layers.Conv2D(channel, size, strides=[1, 1], padding='same',
            #                               kernel_initializer=initializer, use_bias=False)(skip1)
            # x = tf.keras.layers.Concatenate()([x, skip1])

    x = tf.keras.layers.Conv3DTranspose(1, 4, strides=[2,2,1],
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=x)

def Generator_lstm_video_frame2_NOAFS():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs3 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs4 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    initializer = tf.random_normal_initializer(0., 0.02)

    # def downconv(inputs, channels, sizes, strides):
    #     x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
    #                                kernel_initializer=initializer, use_bias=False)(inputs)
    #     x = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(x))
    #     # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))
    #
    #     return x

    def downconv(inputs, channels, sizes, strides, Batch):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        if Batch is True:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        return x

    def upconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3DTranspose(channels, sizes, strides=strides,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(inputs)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x1 = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs1)
        x1 = tf.keras.layers.ReLU()(x1)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(inputs2)
        x = tf.math.multiply(x1, x2_m)
        x = tf.math.add(x, x2_a)
        return x


    def spatial_encoder1(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 64, 4, [2, 2, 2], True)  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2], True)  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 2], True)  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2], True)  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1], True)  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1], True)  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1], True)  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1], True)  # [1, 1, 1]
        skips.append(x)
        skips = reversed(skips[:-1])
        return x, skips

    with tf.compat.v1.variable_scope('scope1', reuse=True):
        conved_output1_1, skips1_1 = spatial_encoder1(inputs1)
        conved_output1_2, skips1_2 = spatial_encoder1(inputs2)
        conved_output1_3, skips1_3 = spatial_encoder1(inputs3)


    lstm_inputs = tf.keras.layers.Concatenate(axis=1)([conved_output1_1, conved_output1_2, conved_output1_3])

    encoder = tf.keras.layers.ConvLSTM2D(filters=512, kernel_size=4, strides=1, padding='same',
                                      data_format=None, dilation_rate=(1, 1), activation='relu',
                                      return_sequences=True, stateful=False)
    decoder = tf.keras.layers.ConvLSTM2D(filters=512, kernel_size=4, strides=1, padding='same',
                                      data_format=None, dilation_rate=(1, 1), activation='relu',
                                      return_sequences=False)
    # encoder.reset_states(states=None)
    encoded_state = encoder(inputs=lstm_inputs)
    decoded_state = decoder(inputs=encoded_state)
    x = decoded_state

    x = tf.expand_dims(x, axis=1)

    with tf.compat.v1.variable_scope('scope2', reuse=True):
        conved_output4, skips4 = spatial_encoder1(inputs4)


    channel_list = [512, 512, 512, 512, 256, 128, 64]
    size_list = [4, 4, 4, 4, 4, 4, 4]
    stride_list = [[2,2,1], [2,2,1], [2,2,1], [2,2,1], [2,2,2], [2,2,2], [2,2,2]]
    skip_list = [True, True, True, True, True, True, True]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for channel, size, stride, drop, skip_d, skip1, skip2, norm in zip(channel_list, size_list, stride_list, drop_list, skip_list, skips1_3, skips4,
                                                        norm_list):
        x = tf.keras.layers.Conv3DTranspose(channel, size, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)
        # x = tf.keras.layers.ReLU()(normalization_layer(x))
        # x = tf.keras.layers.Conv2D(size, 3, strides=1, padding='same',
        #                            kernel_initializer=initializer, use_bias=False)(x)
        # x = normalization_layer(x)
        # res = tf.keras.layers.Conv3D(size, [1, 1, 1], strides=[2, 2, 1], padding='same')(x)
        # x = tf.keras.layers.add([x1, res])
        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            x = tf.keras.layers.BatchNormalization()(x)
            # x1 = spade_norm(x, skip1, channel, 4)

        x = tf.keras.layers.ReLU()(x)
        if skip_d is True:
            # skip1 = tf.keras.layers.Conv2D(channel, size, strides=[1, 1], padding='same',
            #                               kernel_initializer=initializer, use_bias=False)(skip1)
            x = tf.keras.layers.Concatenate()([x, skip1])
        x = spade_norm(x, skip2, channel, 4)

    x = tf.keras.layers.Conv3DTranspose(1, 4, strides=[2,2,2],
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=x)

def Generator_Nolstm_video_frame2_NOAFS():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    initializer = tf.random_normal_initializer(0., 0.02)

    def downconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(x))
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = tf.keras.layers.Conv3D(128, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2 = tf.keras.layers.ReLU()(x2)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(x2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(x2)
        x = tf.math.multiply(inputs1, x2_m)
        x = tf.math.add(x, x2_a)
        return x

    def spatial_encoder(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 64, 4, [2, 2, 2])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 2])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [1, 1, 1]
        skips.append(x)

        skips = reversed(skips[:-1])

        return x, skips

    conved_output1, skips1 = spatial_encoder(inputs1)
    conved_output2, skips2 = spatial_encoder(inputs2)

    x = conved_output1

    channel_list = [512, 512, 512, 512, 256, 128, 64]
    size_list = [4, 4, 4, 4, 4, 4, 4]
    stride_list = [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    skip_list = [True, True, True, True, True, True, True]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for channel, size, stride, drop, skip_d, skip1, skip2, norm in zip(channel_list, size_list, stride_list, drop_list,
                                                                       skip_list, skips1, skips2,
                                                                       norm_list):
        x = tf.keras.layers.Conv3DTranspose(channel, size, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)
        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            x = tf.keras.layers.BatchNormalization()(x)
            # res = tf.keras.layers.Conv3D(channel, 4, strides=1, padding='same',
            #                              kernel_initializer=initializer, use_bias=False)(x)
            # res = tf.keras.layers.ReLU()(res)
            # x = spade_norm(x, skip1, channel, 4)
            res = spade_norm(x, skip2, channel, 4)
            # x = tf.keras.layers.Conv3D(channel, 4, strides=1, padding='same',
            #                            kernel_initializer=initializer, use_bias=False)(x)
            # x = tf.keras.layers.ReLU()(x)
            #
            # # x1 = spade_norm(x, skip1, channel, 4)
            # # x2 = spade_norm(x1, skip2, channel, 4)
            x = tf.keras.layers.Add()([x, res])

        x = tf.keras.layers.ReLU()(x)
        if skip_d is True:
            x = tf.keras.layers.Concatenate()([x, skip1])

    x = tf.keras.layers.Conv3DTranspose(1, 4, strides=[2, 2, 2],
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=[inputs1, inputs2], outputs=x)

def Generator_Nolstm_video_NOAFS_pixguide():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1])
    inputs3 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    initializer = tf.random_normal_initializer(0., 0.02)

    def downconv(inputs, channels, sizes, strides):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        x = tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(x))
        # x = tf.keras.layers.ReLU()(BatchInstanceNorm.BatchInstanceNormalization()(x))

        return x

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = tf.keras.layers.Conv3D(128, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2 = tf.keras.layers.ReLU()(x2)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(x2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(x2)
        x = tf.math.multiply(inputs1, x2_m)
        x = tf.math.add(x, x2_a)
        return x

    def spatial_encoder(inputs1):

        # Downsampling through the model
        skips = []
        # x = tf.keras.layers.Concatenate()([inputs1, inputs2, inputs3])
        x = inputs1
        x = downconv(x, 64, 4, [2, 2, 2])  # [128, 128, 16]
        skips.append(x)
        x = downconv(x, 128, 4, [2, 2, 2])  # [64, 64, 8]
        skips.append(x)
        x = downconv(x, 256, 4, [2, 2, 2])  # [32, 32, 8]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 2])  # [16, 16, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [8, 8, 4]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [4, 4, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [2, 2, 2]
        skips.append(x)
        x = downconv(x, 512, 4, [2, 2, 1])  # [1, 1, 1]
        skips.append(x)

        skips = reversed(skips[:-1])

        return x, skips

    conved_output1, skips1 = spatial_encoder(inputs1)
    conved_output2, skips2 = spatial_encoder(inputs2)
    conved_output3, skips3 = spatial_encoder(inputs3)

    x = conved_output1

    channel_list = [512, 512, 512, 512, 256, 128, 64]
    size_list = [4, 4, 4, 4, 4, 4, 4]
    stride_list = [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    skip_list = [True, True, True, True, True, True, True]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for channel, size, stride, drop, skip_d, skip1, skip2, skip3, norm in zip(channel_list, size_list, stride_list, drop_list,
                                                                       skip_list, skips1, skips2, skips3,
                                                                       norm_list):
        x = tf.keras.layers.Conv3DTranspose(channel, size, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)
        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            ini = tf.keras.layers.BatchNormalization()(x)
            # res = tf.keras.layers.Conv3D(channel, 4, strides=1, padding='same',
            #                              kernel_initializer=initializer, use_bias=False)(x)
            # res = tf.keras.layers.ReLU()(res)
            # x = spade_norm(x, skip1, channel, 4)
            res = spade_norm(ini, skip2, channel, 4)
            x = tf.keras.layers.Add()([ini, res])
            res = spade_norm(x, skip3, channel, 4)
            x = tf.keras.layers.Add()([x, res])
            x = tf.keras.layers.Add()([ini, x])

        x = tf.keras.layers.ReLU()(x)
        if skip_d is True:
            x = tf.keras.layers.Concatenate()([x, skip1])

    x = tf.keras.layers.Conv3DTranspose(1, 4, strides=[2, 2, 2],
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, activation='tanh')(x)

    return tf.keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=x)

def DiscriminatorA1():
  tar = tf.keras.layers.Input(shape=[256, 256, 16, 1], name='target_image')

  x = tar

  initializer = tf.random_normal_initializer(0., 0.02)

  def downsample(inputs, filters, size, apply_batchnorm=True):
      initializer = tf.random_normal_initializer(0., 0.02)

      x = tf.keras.layers.Conv3D(filters, size, strides=[2, 2, 1], padding='same',
                                 kernel_initializer=initializer, use_bias=False)(inputs)
      res = tf.keras.layers.Conv3D(filters, 1, strides=[2, 2, 1], padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
      x = tf.keras.layers.LeakyReLU()(tf.keras.layers.BatchNormalization()(x))
      x = tf.keras.layers.add([x, res])
      x = tf.keras.layers.LeakyReLU()(x)

      return x

  down1 = downsample(x, 64, 4, False)  # (bs, 128, 128, 64)
  down2 = downsample(down1, 128, 4)  # (bs, 64, 64, 128)
  down3 = downsample(down2, 256, 4)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv3D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv3D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
  return tf.keras.Model(inputs=[tar], outputs=[last, down1, down2, down3, last])

def DiscriminatorA2(sample_z, sample_t):
  inp = tf.keras.layers.Input(shape=[256, 256, sample_z, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, sample_z, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

  initializer = tf.random_normal_initializer(0., 0.02)

  def downconv(inputs, filters, size, apply_batchnorm=True):
      x = tf.keras.layers.Conv3D(filters, size, strides=[2, 2, 1], padding='same',
                                 kernel_initializer=initializer, use_bias=False)(inputs)
      if apply_batchnorm is True:
          x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.LeakyReLU()(x)
      return x

  down1 = downconv(x, 64, 4, False)  # (bs, 128, 128, 64)
  down2 = downconv(down1, 128, 4)  # (bs, 64, 64, 128)
  down3 = downconv(down2, 256, 4)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv3D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (bs, 33, 33, 512)
  print(zero_pad2.shape)

  last = tf.keras.layers.Conv3D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
  # return tf.keras.Model(inputs=[inp, tar], outputs=last)
  return tf.keras.Model(inputs=[inp, tar], outputs=[last, down1, down2, down3, last])

def DiscriminatorB2():
  inp1 = tf.keras.layers.Input(shape=[256, 256, 16, 1], name='input_image1')
  inp2 = tf.keras.layers.Input(shape=[256, 256, 16, 1], name='input_image2')
  tar1 = tf.keras.layers.Input(shape=[256, 256, 16, 1], name='target_image1')
  tar2 = tf.keras.layers.Input(shape=[256, 256, 16, 1], name='target_image2')

  x = tf.keras.layers.concatenate([inp1, inp2, tar1, tar2])  # (bs, 256, 256, channels*2)
  initializer = tf.random_normal_initializer(0., 0.02)

  def downconv(inputs, filters, size, apply_batchnorm=True):
      x = tf.keras.layers.Conv3D(filters, size, strides=[2, 2, 1], padding='same',
                                 kernel_initializer=initializer, use_bias=False)(inputs)
      if apply_batchnorm is True:
          x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.LeakyReLU()(x)
      return x

  down1 = downconv(x, 64, 4, False)  # (bs, 128, 128, 64)
  down2 = downconv(down1, 128, 4)  # (bs, 64, 64, 128)
  down3 = downconv(down2, 256, 4)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv3D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (bs, 33, 33, 512)
  print(zero_pad2.shape)

  last = tf.keras.layers.Conv3D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  # return tf.keras.Model(inputs=[inp1, inp2, tar1, tar2], outputs=last)
  return tf.keras.Model(inputs=[inp1, inp2, tar1, tar2], outputs=[last, down1, down2, down3, last])



def l1_loss(input, target):
  mae = tf.keras.losses.MeanAbsoluteError()
  loss = mae(input, target)
  return loss

def l2_loss(input, target):
  mse = tf.keras.losses.MeanSquaredError()
  loss = mse(input, target)
  return loss

def cosinesimilarity_loss(input, target):
  cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
  loss = tf.add(cosine_loss(target, input), 1)
  return loss

def ssim_loss(input, target):
  loss = tf.reduce_mean(tf.image.ssim(target, input, max_val=1.0))
  return loss

def binarycrossentroty_loss_G(disc_real_output, disc_generated_output):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  return loss

def binarycrossentroty_loss_D(disc_real_output, disc_generated_output):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

def LS_loss_G(disc_real_output, disc_generated_output):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  loss = 0.5 * tf.reduce_mean(tf.nn.l2_loss(disc_generated_output-1))
  return loss

def LS_loss_D(disc_real_output, disc_generated_output):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  real_loss = 0.5 * tf.reduce_mean(tf.nn.l2_loss(disc_real_output-1))

  generated_loss = 0.5 * tf.reduce_mean(tf.nn.l2_loss(disc_generated_output))

  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

def adversarial_loss_frame(D_real, D_fake):
  D_loss = 0.5 * (tf.reduce_mean((D_real - 1) ** 2) + tf.reduce_mean(D_fake ** 2))
  return D_loss

def adversarial_loss_frame_G(D_real, D_fake):
  D_loss = 0.5 * (tf.reduce_mean((D_fake - 1) ** 2) + tf.reduce_mean(D_real ** 2))
  return D_loss


