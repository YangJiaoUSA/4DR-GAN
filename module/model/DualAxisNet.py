from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import gc
# import BatchInstanceNorm

def Generator_correct_z():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1])

    def conv(inputs, channels, sizes, strides, Batch):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        if Batch is True:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        return x


    initializer = tf.random_normal_initializer(0., 0.02)
    fileter_size = 4
    last = tf.keras.layers.Conv3DTranspose(1, fileter_size,
                                           strides=[2, 2, 1],
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)


    # Downsampling through the model
    skips = []

    x = conv(inputs1, 64, 4, [2, 2, 1], False) #128
    skips.append(x)

    x = conv(x, 128, 4, [2, 2, 1], True)  # 64
    skips.append(x)

    x = conv(x, 256, 4, [2, 2, 1], True)  # 32
    skips.append(x)

    x = conv(x, 512, 4, [2, 2, 1], True)  # 16
    skips.append(x)

    x = conv(x, 512, 4, [2, 2, 1], True)  # 8
    skips.append(x)

    x = conv(x, 512, 4, [2, 2, 1], True)  # 4
    skips.append(x)

    x = conv(x, 512, 4, [2, 2, 1], True)  # 2
    skips.append(x)

    x = conv(x, 512, 4, [2, 2, 1], True)  # 2
    skips.append(x)

    skips = reversed(skips[:-1])
    size_list = [512, 512, 512, 512, 256, 128, 64]
    stride_list = [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1]]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for size, drop, stride, skip, norm in zip(size_list, drop_list, stride_list, skips, norm_list):
        x = tf.keras.layers.Conv3DTranspose(size, 4, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)

        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=[inputs1], outputs=x)

def Generator_correct_t():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 10, 1])

    def conv(inputs, channels, sizes, strides, Batch):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        if Batch is True:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        return x


    initializer = tf.random_normal_initializer(0., 0.02)
    fileter_size = 4
    last = tf.keras.layers.Conv3DTranspose(1, fileter_size,
                                           strides=[2, 2, 1],
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)


    # Downsampling through the model
    skips = []

    x = conv(inputs1, 64, 4, [2, 2, 1], False) #128
    skips.append(x)

    x = conv(x, 128, 4, [2, 2, 1], True)  # 64
    skips.append(x)

    x = conv(x, 256, 4, [2, 2, 1], True)  # 32
    skips.append(x)

    x = conv(x, 512, 4, [2, 2, 1], True)  # 16
    skips.append(x)

    x = conv(x, 512, 4, [2, 2, 1], True)  # 8
    skips.append(x)

    x = conv(x, 512, 4, [2, 2, 1], True)  # 4
    skips.append(x)

    x = conv(x, 512, 4, [2, 2, 1], True)  # 2
    skips.append(x)

    x = conv(x, 512, 4, [2, 2, 1], True)  # 2
    skips.append(x)

    skips = reversed(skips[:-1])
    size_list = [512, 512, 512, 512, 256, 128, 64]
    stride_list = [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1]]
    drop_list = [True, True, True, True, True, True, True]
    norm_list = [True, True, True, True, True, True, True]
    # Upsampling and establishing the skip connections
    for size, drop, stride, skip, norm in zip(size_list, drop_list, stride_list, skips, norm_list):
        x = tf.keras.layers.Conv3DTranspose(size, 4, strides=stride,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(x)

        if drop is True:
            x = tf.keras.layers.Dropout(0.5)(x)
        if norm is True:
            tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=[inputs1], outputs=x)

def DiscriminatorBz():
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

def DiscriminatorBt():
  inp1 = tf.keras.layers.Input(shape=[256, 256, 10, 1], name='input_image1')
  inp2 = tf.keras.layers.Input(shape=[256, 256, 10, 1], name='input_image2')
  tar1 = tf.keras.layers.Input(shape=[256, 256, 10, 1], name='target_image1')
  tar2 = tf.keras.layers.Input(shape=[256, 256, 10, 1], name='target_image2')

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

def Generator_correct_connect():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1, 10])

    def conv(inputs, channels, sizes, strides, Batch):
        x = tf.keras.layers.Conv3D(channels, sizes, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs)
        if Batch is True:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        return x

    def encoder(inputs):
        skips = []
        x = conv(inputs, 32, 4, [2, 2, 1], False)  # 128
        skips.append(x)

        x = conv(x, 64, 4, [2, 2, 1], True)  # 64
        skips.append(x)

        x = conv(x, 128, 4, [2, 2, 1], True)  # 32
        skips.append(x)

        x = conv(x, 256, 4, [2, 2, 1], True)  # 16
        skips.append(x)

        x = conv(x, 256, 4, [2, 2, 1], True)  # 8
        skips.append(x)

        x = conv(x, 256, 4, [2, 2, 1], True)  # 4
        skips.append(x)

        x = conv(x, 256, 4, [2, 2, 1], True)  # 2
        skips.append(x)

        x = conv(x, 256, 4, [2, 2, 1], True)  # 2
        # skips.append(x)
        skips = reversed(skips)
        return x, skips

    def decoder(inputs, skips):
        size_list = [256, 256, 256, 256, 128, 64, 32]
        stride_list = [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1]]
        drop_list = [True, True, True, True, True, True, True]
        norm_list = [True, True, True, True, True, True, True]
        # Upsampling and establishing the skip connections
        x = inputs
        for size, drop, stride, skip, norm in zip(size_list, drop_list, stride_list, skips, norm_list):
            x = tf.keras.layers.Conv3DTranspose(size, 4, strides=stride,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                use_bias=False)(x)

            if drop is True:
                x = tf.keras.layers.Dropout(0.5)(x)
            if norm is True:
                tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return x

    def decoder2(inputs, skips2):
        size_list = [256, 256, 256, 256, 128, 64, 32]
        stride_list = [[2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1]]
        drop_list = [True, True, True, True, True, True, True]
        norm_list = [True, True, True, True, True, True, True]
        # Upsampling and establishing the skip connections
        x = inputs
        for size, drop, stride, skip2, norm in zip(size_list, drop_list, stride_list, skips2, norm_list):
            x = tf.keras.layers.Conv3DTranspose(size, 4, strides=stride,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                use_bias=False)(x)

            if drop is True:
                x = tf.keras.layers.Dropout(0.5)(x)
            if norm is True:
                tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Concatenate()([x, skip2])

        x = last(x)

        return x


    initializer = tf.random_normal_initializer(0., 0.02)
    fileter_size = 4
    last = tf.keras.layers.Conv3DTranspose(1, fileter_size,
                                           strides=[2, 2, 1],
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)


    # Downsampling through the model
    input_t_list = []
    spatial_out_list = []
    skips1_list = []
    conved_output1_list = []

    with tf.compat.v1.variable_scope('scope1', reuse=True):
        for t in range(0, inputs1.shape[5]):
            conved_output1, skips1_1 = encoder((inputs1[:,:,:,:,:,t]))
            skips1_list.append(skips1_1)
            conved_output1_list.append(conved_output1)

    with tf.compat.v1.variable_scope('scope2', reuse=True):
        for conved_output1, skips1 in zip(conved_output1_list, skips1_list):
            spatial_out = decoder(conved_output1, skips1)
            spatial_out_list.append(spatial_out)

    spatial_out_list = tf.convert_to_tensor(spatial_out_list)
    spatial_out_list = tf.transpose(spatial_out_list, [1, 2, 3, 4, 5, 0])
    inputs2 = tf.concat([tf.transpose(spatial_out_list, [0,1,2,5,4,3]), tf.transpose(inputs1, [0,1,2,5,4,3])], axis=4)

    skips2_list = []
    conved_output2_list = []
    temporal_out_list = []
    with tf.compat.v1.variable_scope('scope3', reuse=True):
        for z in range(0, inputs2.shape[5]):
            conved_output2, skips2 = encoder((inputs2[:, :, :, :, :, z]))
            skips2_list.append(skips2)
            conved_output2_list.append(conved_output2)

    with tf.compat.v1.variable_scope('scope4', reuse=True):
        for conved_output2, skips2 in zip(conved_output2_list, skips2_list):
            temporal_out = decoder2(conved_output2, skips2)
            temporal_out_list.append(temporal_out)

    temporal_out_list = tf.convert_to_tensor(temporal_out_list)
    temporal_out_list = tf.transpose(temporal_out_list, [1, 2, 3, 0, 5, 4])

    return tf.keras.Model(inputs=[inputs1], outputs=[spatial_out_list,temporal_out_list])

def Generator_correct_connect_reuse():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1, 10])
    initializer = tf.random_normal_initializer(0., 0.02)

    conv1_1 = tf.keras.layers.Conv3D(32, 4, strides=[2,2,1], padding='same',
                                   kernel_initializer=initializer, use_bias=False)
    relu1_1 = tf.keras.layers.ReLU()

    conv1_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_2 = tf.keras.layers.BatchNormalization()
    relu1_2 = tf.keras.layers.ReLU()

    conv1_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_3 = tf.keras.layers.BatchNormalization()
    relu1_3 = tf.keras.layers.ReLU()

    conv1_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_4 = tf.keras.layers.BatchNormalization()
    relu1_4 = tf.keras.layers.ReLU()

    conv1_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_5 = tf.keras.layers.BatchNormalization()
    relu1_5 = tf.keras.layers.ReLU()

    conv1_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_6 = tf.keras.layers.BatchNormalization()
    relu1_6 = tf.keras.layers.ReLU()

    conv1_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_7 = tf.keras.layers.BatchNormalization()
    relu1_7 = tf.keras.layers.ReLU()

    conv1_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_8 = tf.keras.layers.BatchNormalization()
    relu1_8 = tf.keras.layers.ReLU()
    
    
    #--------------------------------------------------------------------------------------------------------------
    conv2_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu2_1 = tf.keras.layers.ReLU()

    conv2_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_2 = tf.keras.layers.BatchNormalization()
    relu2_2 = tf.keras.layers.ReLU()

    conv2_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_3 = tf.keras.layers.BatchNormalization()
    relu2_3 = tf.keras.layers.ReLU()

    conv2_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_4 = tf.keras.layers.BatchNormalization()
    relu2_4 = tf.keras.layers.ReLU()

    conv2_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_5 = tf.keras.layers.BatchNormalization()
    relu2_5 = tf.keras.layers.ReLU()

    conv2_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_6 = tf.keras.layers.BatchNormalization()
    relu2_6 = tf.keras.layers.ReLU()

    conv2_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_7 = tf.keras.layers.BatchNormalization()
    relu2_7 = tf.keras.layers.ReLU()

    conv2_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_8 = tf.keras.layers.BatchNormalization()
    relu2_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv_trans3_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2,2,1],
                                                padding='same',
                                                kernel_initializer=initializer,
                                                use_bias=False)
    drop3_1 = tf.keras.layers.Dropout(0.5)
    batchnorm3_1 = tf.keras.layers.BatchNormalization()
    relu3_1 = tf.keras.layers.ReLU()
    cat3_1 = tf.keras.layers.Concatenate()

    conv_trans3_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_2 = tf.keras.layers.Dropout(0.5)
    batchnorm3_2 = tf.keras.layers.BatchNormalization()
    relu3_2 = tf.keras.layers.ReLU()
    cat3_2 = tf.keras.layers.Concatenate()

    conv_trans3_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_3 = tf.keras.layers.Dropout(0.5)
    batchnorm3_3 = tf.keras.layers.BatchNormalization()
    relu3_3 = tf.keras.layers.ReLU()
    cat3_3 = tf.keras.layers.Concatenate()

    conv_trans3_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_4 = tf.keras.layers.Dropout(0.5)
    batchnorm3_4 = tf.keras.layers.BatchNormalization()
    relu3_4 = tf.keras.layers.ReLU()
    cat3_4 = tf.keras.layers.Concatenate()

    conv_trans3_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_5 = tf.keras.layers.Dropout(0.5)
    batchnorm3_5 = tf.keras.layers.BatchNormalization()
    relu3_5 = tf.keras.layers.ReLU()
    cat3_5 = tf.keras.layers.Concatenate()

    conv_trans3_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_6 = tf.keras.layers.Dropout(0.5)
    batchnorm3_6 = tf.keras.layers.BatchNormalization()
    relu3_6 = tf.keras.layers.ReLU()
    cat3_6 = tf.keras.layers.Concatenate()

    conv_trans3_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_7 = tf.keras.layers.Dropout(0.5)
    batchnorm3_7 = tf.keras.layers.BatchNormalization()
    relu3_7 = tf.keras.layers.ReLU()
    cat3_7 = tf.keras.layers.Concatenate()

    conv_trans3_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                           strides=[2,2,1],
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    # --------------------------------------------------------------------------------------------------------------
    conv_trans4_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_1 = tf.keras.layers.Dropout(0.5)
    batchnorm4_1 = tf.keras.layers.BatchNormalization()
    relu4_1 = tf.keras.layers.ReLU()
    cat4_1 = tf.keras.layers.Concatenate()

    conv_trans4_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_2 = tf.keras.layers.Dropout(0.5)
    batchnorm4_2 = tf.keras.layers.BatchNormalization()
    relu4_2 = tf.keras.layers.ReLU()
    cat4_2 = tf.keras.layers.Concatenate()

    conv_trans4_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_3 = tf.keras.layers.Dropout(0.5)
    batchnorm4_3 = tf.keras.layers.BatchNormalization()
    relu4_3 = tf.keras.layers.ReLU()
    cat4_3 = tf.keras.layers.Concatenate()

    conv_trans4_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_4 = tf.keras.layers.Dropout(0.5)
    batchnorm4_4 = tf.keras.layers.BatchNormalization()
    relu4_4 = tf.keras.layers.ReLU()
    cat4_4 = tf.keras.layers.Concatenate()

    conv_trans4_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_5 = tf.keras.layers.Dropout(0.5)
    batchnorm4_5 = tf.keras.layers.BatchNormalization()
    relu4_5 = tf.keras.layers.ReLU()
    cat4_5 = tf.keras.layers.Concatenate()

    conv_trans4_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_6 = tf.keras.layers.Dropout(0.5)
    batchnorm4_6 = tf.keras.layers.BatchNormalization()
    relu4_6 = tf.keras.layers.ReLU()
    cat4_6 = tf.keras.layers.Concatenate()

    conv_trans4_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_7 = tf.keras.layers.Dropout(0.5)
    batchnorm4_7 = tf.keras.layers.BatchNormalization()
    relu4_7 = tf.keras.layers.ReLU()
    cat4_7 = tf.keras.layers.Concatenate()

    conv_trans4_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

    #-------------------------------------------------------------------------------
    def encoder1(inputs):
        skips = []
        x = conv1_1(inputs)
        x = relu1_1(x)
        skips.append(x)

        x = conv1_2(x)
        x = batchnorm1_2(x)
        x = relu1_2(x)
        skips.append(x)

        x = conv1_3(x)
        x = batchnorm1_3(x)
        x = relu1_3(x)
        skips.append(x)

        x = conv1_4(x)
        x = batchnorm1_4(x)
        x = relu1_4(x)
        skips.append(x)

        x = conv1_5(x)
        x = batchnorm1_5(x)
        x = relu1_5(x)
        skips.append(x)

        x = conv1_6(x)
        x = batchnorm1_6(x)
        x = relu1_6(x)
        skips.append(x)

        x = conv1_7(x)
        x = batchnorm1_7(x)
        x = relu1_7(x)
        skips.append(x)

        x = conv1_8(x)
        x = batchnorm1_8(x)
        x = relu1_8(x)

        return x, skips
    
    def encoder2(inputs):
        skips = []
        x = conv2_1(inputs)
        x = relu2_1(x)
        skips.append(x)

        x = conv2_2(x)
        x = batchnorm2_2(x)
        x = relu2_2(x)
        skips.append(x)

        x = conv2_3(x)
        x = batchnorm2_3(x)
        x = relu2_3(x)
        skips.append(x)

        x = conv2_4(x)
        x = batchnorm2_4(x)
        x = relu2_4(x)
        skips.append(x)

        x = conv2_5(x)
        x = batchnorm2_5(x)
        x = relu2_5(x)
        skips.append(x)

        x = conv2_6(x)
        x = batchnorm2_6(x)
        x = relu2_6(x)
        skips.append(x)

        x = conv2_7(x)
        x = batchnorm2_7(x)
        x = relu2_7(x)
        skips.append(x)

        x = conv2_8(x)
        x = batchnorm2_8(x)
        x = relu2_8(x)

        return x, skips

    def decoder1(inputs, skips):
        x = conv_trans3_1(inputs)
        x = batchnorm3_1(x)
        x = drop3_1(x)
        x = relu3_1(x)
        x = cat3_1([x, skips[6]])

        x = conv_trans3_2(x)
        x = batchnorm3_2(x)
        x = drop3_2(x)
        x = relu3_2(x)
        x = cat3_2([x, skips[5]])

        x = conv_trans3_3(x)
        x = batchnorm3_3(x)
        x = drop3_3(x)
        x = relu3_3(x)
        x = cat3_3([x, skips[4]])

        x = conv_trans3_4(x)
        x = batchnorm3_4(x)
        x = relu3_4(x)
        x = cat3_4([x, skips[3]])

        x = conv_trans3_5(x)
        x = batchnorm3_5(x)
        x = relu3_5(x)
        x = cat3_5([x, skips[2]])

        x = conv_trans3_6(x)
        x = batchnorm3_6(x)
        x = relu3_6(x)
        x = cat3_6([x, skips[1]])

        x = conv_trans3_7(x)
        x = batchnorm3_7(x)
        x = relu3_7(x)
        x = cat3_7([x, skips[0]])

        x = conv_trans3_last(x)
        return x

    def decoder2(inputs, skips):
        x = conv_trans4_1(inputs)
        x = batchnorm4_1(x)
        x = drop4_1(x)
        x = relu4_1(x)
        x = cat4_1([x, skips[6]])

        x = conv_trans4_2(x)
        x = batchnorm4_2(x)
        x = drop4_2(x)
        x = relu4_2(x)
        x = cat4_2([x, skips[5]])

        x = conv_trans4_3(x)
        x = batchnorm4_3(x)
        x = drop4_3(x)
        x = relu4_3(x)
        x = cat4_3([x, skips[4]])

        x = conv_trans4_4(x)
        x = batchnorm4_4(x)
        x = relu4_4(x)
        x = cat4_4([x, skips[3]])

        x = conv_trans4_5(x)
        x = batchnorm4_5(x)
        x = relu4_5(x)
        x = cat4_5([x, skips[2]])

        x = conv_trans4_6(x)
        x = batchnorm4_6(x)
        x = relu4_6(x)
        x = cat4_6([x, skips[1]])

        x = conv_trans4_7(x)
        x = batchnorm4_7(x)
        x = relu4_7(x)
        x = cat4_7([x, skips[0]])

        x = conv_trans4_last(x)
        return x



    # Downsampling through the model
    spatial_out_list = []
    skips1_list = []
    conved_output1_list = []

    # with tf.compat.v1.variable_scope('scope1', reuse=True):
    for t in range(0, inputs1.shape[5]):
        conved_output1, skips1_1 = encoder1((inputs1[:,:,:,:,:,t]))
        skips1_list.append(skips1_1)
        conved_output1_list.append(conved_output1)

    # with tf.compat.v1.variable_scope('scope2', reuse=True):
    for conved_output1, skips1 in zip(conved_output1_list, skips1_list):
        spatial_out = decoder1(conved_output1, skips1)
        spatial_out_list.append(spatial_out)

    spatial_out_list = tf.convert_to_tensor(spatial_out_list)
    spatial_out_list = tf.transpose(spatial_out_list, [1, 2, 3, 4, 5, 0])
    inputs2 = tf.concat([tf.transpose(spatial_out_list, [0,1,2,5,4,3]), tf.transpose(inputs1, [0,1,2,5,4,3])], axis=4)


    skips2_list = []
    conved_output2_list = []
    temporal_out_list = []
    # with tf.compat.v1.variable_scope('scope3', reuse=True):
    for z in range(0, inputs2.shape[5]):
        conved_output2, skips2 = encoder2((inputs2[:, :, :, :, :, z]))
        skips2_list.append(skips2)
        conved_output2_list.append(conved_output2)

    # with tf.compat.v1.variable_scope('scope4', reuse=True):
    for conved_output2, skips2 in zip(conved_output2_list, skips2_list):
        temporal_out = decoder2(conved_output2, skips2)
        temporal_out_list.append(temporal_out)

    temporal_out_list = tf.convert_to_tensor(temporal_out_list)
    temporal_out_list = tf.transpose(temporal_out_list, [1, 2, 3, 0, 5, 4])

    return tf.keras.Model(inputs=[inputs1], outputs=[spatial_out_list,temporal_out_list])


def Generator_correct_connect_reuse_2input():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1, 10])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1, 10])

    initializer = tf.random_normal_initializer(0., 0.02)

    conv1_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu1_1 = tf.keras.layers.ReLU()

    conv1_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_2 = tf.keras.layers.BatchNormalization()
    relu1_2 = tf.keras.layers.ReLU()

    conv1_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_3 = tf.keras.layers.BatchNormalization()
    relu1_3 = tf.keras.layers.ReLU()

    conv1_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_4 = tf.keras.layers.BatchNormalization()
    relu1_4 = tf.keras.layers.ReLU()

    conv1_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_5 = tf.keras.layers.BatchNormalization()
    relu1_5 = tf.keras.layers.ReLU()

    conv1_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_6 = tf.keras.layers.BatchNormalization()
    relu1_6 = tf.keras.layers.ReLU()

    conv1_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_7 = tf.keras.layers.BatchNormalization()
    relu1_7 = tf.keras.layers.ReLU()

    conv1_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_8 = tf.keras.layers.BatchNormalization()
    relu1_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv2_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu2_1 = tf.keras.layers.ReLU()

    conv2_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_2 = tf.keras.layers.BatchNormalization()
    relu2_2 = tf.keras.layers.ReLU()

    conv2_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_3 = tf.keras.layers.BatchNormalization()
    relu2_3 = tf.keras.layers.ReLU()

    conv2_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_4 = tf.keras.layers.BatchNormalization()
    relu2_4 = tf.keras.layers.ReLU()

    conv2_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_5 = tf.keras.layers.BatchNormalization()
    relu2_5 = tf.keras.layers.ReLU()

    conv2_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_6 = tf.keras.layers.BatchNormalization()
    relu2_6 = tf.keras.layers.ReLU()

    conv2_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_7 = tf.keras.layers.BatchNormalization()
    relu2_7 = tf.keras.layers.ReLU()

    conv2_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_8 = tf.keras.layers.BatchNormalization()
    relu2_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv_trans3_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_1 = tf.keras.layers.Dropout(0.5)
    batchnorm3_1 = tf.keras.layers.BatchNormalization()
    relu3_1 = tf.keras.layers.ReLU()
    cat3_1 = tf.keras.layers.Concatenate()

    conv_trans3_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_2 = tf.keras.layers.Dropout(0.5)
    batchnorm3_2 = tf.keras.layers.BatchNormalization()
    relu3_2 = tf.keras.layers.ReLU()
    cat3_2 = tf.keras.layers.Concatenate()

    conv_trans3_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_3 = tf.keras.layers.Dropout(0.5)
    batchnorm3_3 = tf.keras.layers.BatchNormalization()
    relu3_3 = tf.keras.layers.ReLU()
    cat3_3 = tf.keras.layers.Concatenate()

    conv_trans3_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_4 = tf.keras.layers.Dropout(0.5)
    batchnorm3_4 = tf.keras.layers.BatchNormalization()
    relu3_4 = tf.keras.layers.ReLU()
    cat3_4 = tf.keras.layers.Concatenate()

    conv_trans3_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_5 = tf.keras.layers.Dropout(0.5)
    batchnorm3_5 = tf.keras.layers.BatchNormalization()
    relu3_5 = tf.keras.layers.ReLU()
    cat3_5 = tf.keras.layers.Concatenate()

    conv_trans3_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_6 = tf.keras.layers.Dropout(0.5)
    batchnorm3_6 = tf.keras.layers.BatchNormalization()
    relu3_6 = tf.keras.layers.ReLU()
    cat3_6 = tf.keras.layers.Concatenate()

    conv_trans3_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_7 = tf.keras.layers.Dropout(0.5)
    batchnorm3_7 = tf.keras.layers.BatchNormalization()
    relu3_7 = tf.keras.layers.ReLU()
    cat3_7 = tf.keras.layers.Concatenate()

    conv_trans3_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

    # --------------------------------------------------------------------------------------------------------------
    conv_trans4_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_1 = tf.keras.layers.Dropout(0.5)
    batchnorm4_1 = tf.keras.layers.BatchNormalization()
    relu4_1 = tf.keras.layers.ReLU()
    cat4_1 = tf.keras.layers.Concatenate()

    conv_trans4_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_2 = tf.keras.layers.Dropout(0.5)
    batchnorm4_2 = tf.keras.layers.BatchNormalization()
    relu4_2 = tf.keras.layers.ReLU()
    cat4_2 = tf.keras.layers.Concatenate()

    conv_trans4_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_3 = tf.keras.layers.Dropout(0.5)
    batchnorm4_3 = tf.keras.layers.BatchNormalization()
    relu4_3 = tf.keras.layers.ReLU()
    cat4_3 = tf.keras.layers.Concatenate()

    conv_trans4_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_4 = tf.keras.layers.Dropout(0.5)
    batchnorm4_4 = tf.keras.layers.BatchNormalization()
    relu4_4 = tf.keras.layers.ReLU()
    cat4_4 = tf.keras.layers.Concatenate()

    conv_trans4_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_5 = tf.keras.layers.Dropout(0.5)
    batchnorm4_5 = tf.keras.layers.BatchNormalization()
    relu4_5 = tf.keras.layers.ReLU()
    cat4_5 = tf.keras.layers.Concatenate()

    conv_trans4_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_6 = tf.keras.layers.Dropout(0.5)
    batchnorm4_6 = tf.keras.layers.BatchNormalization()
    relu4_6 = tf.keras.layers.ReLU()
    cat4_6 = tf.keras.layers.Concatenate()

    conv_trans4_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_7 = tf.keras.layers.Dropout(0.5)
    batchnorm4_7 = tf.keras.layers.BatchNormalization()
    relu4_7 = tf.keras.layers.ReLU()
    cat4_7 = tf.keras.layers.Concatenate()

    conv_trans4_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

    # -------------------------------------------------------------------------------
    def encoder1(inputs):
        skips = []
        x = conv1_1(inputs)
        x = relu1_1(x)
        skips.append(x)

        x = conv1_2(x)
        x = batchnorm1_2(x)
        x = relu1_2(x)
        skips.append(x)

        x = conv1_3(x)
        x = batchnorm1_3(x)
        x = relu1_3(x)
        skips.append(x)

        x = conv1_4(x)
        x = batchnorm1_4(x)
        x = relu1_4(x)
        skips.append(x)

        x = conv1_5(x)
        x = batchnorm1_5(x)
        x = relu1_5(x)
        skips.append(x)

        x = conv1_6(x)
        x = batchnorm1_6(x)
        x = relu1_6(x)
        skips.append(x)

        x = conv1_7(x)
        x = batchnorm1_7(x)
        x = relu1_7(x)
        skips.append(x)

        x = conv1_8(x)
        x = batchnorm1_8(x)
        x = relu1_8(x)

        return x, skips

    def encoder2(inputs):
        skips = []
        x = conv2_1(inputs)
        x = relu2_1(x)
        skips.append(x)

        x = conv2_2(x)
        x = batchnorm2_2(x)
        x = relu2_2(x)
        skips.append(x)

        x = conv2_3(x)
        x = batchnorm2_3(x)
        x = relu2_3(x)
        skips.append(x)

        x = conv2_4(x)
        x = batchnorm2_4(x)
        x = relu2_4(x)
        skips.append(x)

        x = conv2_5(x)
        x = batchnorm2_5(x)
        x = relu2_5(x)
        skips.append(x)

        x = conv2_6(x)
        x = batchnorm2_6(x)
        x = relu2_6(x)
        skips.append(x)

        x = conv2_7(x)
        x = batchnorm2_7(x)
        x = relu2_7(x)
        skips.append(x)

        x = conv2_8(x)
        x = batchnorm2_8(x)
        x = relu2_8(x)

        return x, skips

    def decoder1(inputs, skips):
        x = conv_trans3_1(inputs)
        x = batchnorm3_1(x)
        x = drop3_1(x)
        x = relu3_1(x)
        x = cat3_1([x, skips[6]])

        x = conv_trans3_2(x)
        x = batchnorm3_2(x)
        x = drop3_2(x)
        x = relu3_2(x)
        x = cat3_2([x, skips[5]])

        x = conv_trans3_3(x)
        x = batchnorm3_3(x)
        x = drop3_3(x)
        x = relu3_3(x)
        x = cat3_3([x, skips[4]])

        x = conv_trans3_4(x)
        x = batchnorm3_4(x)
        x = relu3_4(x)
        x = cat3_4([x, skips[3]])

        x = conv_trans3_5(x)
        x = batchnorm3_5(x)
        x = relu3_5(x)
        x = cat3_5([x, skips[2]])

        x = conv_trans3_6(x)
        x = batchnorm3_6(x)
        x = relu3_6(x)
        x = cat3_6([x, skips[1]])

        x = conv_trans3_7(x)
        x = batchnorm3_7(x)
        x = relu3_7(x)
        x = cat3_7([x, skips[0]])

        x = conv_trans3_last(x)
        return x

    def decoder2(inputs, skips):
        x = conv_trans4_1(inputs)
        x = batchnorm4_1(x)
        x = drop4_1(x)
        x = relu4_1(x)
        x = cat4_1([x, skips[6]])

        x = conv_trans4_2(x)
        x = batchnorm4_2(x)
        x = drop4_2(x)
        x = relu4_2(x)
        x = cat4_2([x, skips[5]])

        x = conv_trans4_3(x)
        x = batchnorm4_3(x)
        x = drop4_3(x)
        x = relu4_3(x)
        x = cat4_3([x, skips[4]])

        x = conv_trans4_4(x)
        x = batchnorm4_4(x)
        x = relu4_4(x)
        x = cat4_4([x, skips[3]])

        x = conv_trans4_5(x)
        x = batchnorm4_5(x)
        x = relu4_5(x)
        x = cat4_5([x, skips[2]])

        x = conv_trans4_6(x)
        x = batchnorm4_6(x)
        x = relu4_6(x)
        x = cat4_6([x, skips[1]])

        x = conv_trans4_7(x)
        x = batchnorm4_7(x)
        x = relu4_7(x)
        x = cat4_7([x, skips[0]])

        x = conv_trans4_last(x)
        return x

    # Downsampling through the model
    spatial_out_list = []
    skips1_list = []
    conved_output1_list = []

    concat_inputs1 = tf.concat([inputs1, inputs2],
                        axis=4)
    # with tf.compat.v1.variable_scope('scope1', reuse=True):
    for t in range(0, concat_inputs1.shape[5]):
        conved_output1, skips1_1 = encoder1((concat_inputs1[:, :, :, :, :, t]))
        skips1_list.append(skips1_1)
        conved_output1_list.append(conved_output1)

    # with tf.compat.v1.variable_scope('scope2', reuse=True):
    for conved_output1, skips1 in zip(conved_output1_list, skips1_list):
        spatial_out = decoder1(conved_output1, skips1)
        spatial_out_list.append(spatial_out)

    spatial_out_list = tf.convert_to_tensor(spatial_out_list)
    spatial_out_list = tf.transpose(spatial_out_list, [1, 2, 3, 4, 5, 0])
    concat_inputs2 = tf.concat([tf.transpose(spatial_out_list, [0, 1, 2, 5, 4, 3]), tf.transpose(inputs1, [0, 1, 2, 5, 4, 3]), tf.transpose(inputs2, [0, 1, 2, 5, 4, 3])],
                        axis=4)

    skips2_list = []
    conved_output2_list = []
    temporal_out_list = []
    # with tf.compat.v1.variable_scope('scope3', reuse=True):
    for z in range(0, concat_inputs2.shape[5]):
        conved_output2, skips2 = encoder2((concat_inputs2[:, :, :, :, :, z]))
        skips2_list.append(skips2)
        conved_output2_list.append(conved_output2)

    # with tf.compat.v1.variable_scope('scope4', reuse=True):
    for conved_output2, skips2 in zip(conved_output2_list, skips2_list):
        temporal_out = decoder2(conved_output2, skips2)
        temporal_out_list.append(temporal_out)

    temporal_out_list = tf.convert_to_tensor(temporal_out_list)
    temporal_out_list = tf.transpose(temporal_out_list, [1, 2, 3, 0, 5, 4])

    return tf.keras.Model(inputs=[inputs1, inputs2], outputs=[spatial_out_list, temporal_out_list])

def Generator_correct_connect_reuse_ref():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1, 10])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1, 10])
    initializer = tf.random_normal_initializer(0., 0.02)

    conv1_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu1_1 = tf.keras.layers.ReLU()

    conv1_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_2 = tf.keras.layers.BatchNormalization()
    relu1_2 = tf.keras.layers.ReLU()

    conv1_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_3 = tf.keras.layers.BatchNormalization()
    relu1_3 = tf.keras.layers.ReLU()

    conv1_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_4 = tf.keras.layers.BatchNormalization()
    relu1_4 = tf.keras.layers.ReLU()

    conv1_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_5 = tf.keras.layers.BatchNormalization()
    relu1_5 = tf.keras.layers.ReLU()

    conv1_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_6 = tf.keras.layers.BatchNormalization()
    relu1_6 = tf.keras.layers.ReLU()

    conv1_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_7 = tf.keras.layers.BatchNormalization()
    relu1_7 = tf.keras.layers.ReLU()

    conv1_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_8 = tf.keras.layers.BatchNormalization()
    relu1_8 = tf.keras.layers.ReLU()

    #------------------------------------------------------------------------------------------------------------
    conv5_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu5_1 = tf.keras.layers.ReLU()

    conv5_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm5_2 = tf.keras.layers.BatchNormalization()
    relu5_2 = tf.keras.layers.ReLU()

    conv5_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm5_3 = tf.keras.layers.BatchNormalization()
    relu5_3 = tf.keras.layers.ReLU()

    conv5_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm5_4 = tf.keras.layers.BatchNormalization()
    relu5_4 = tf.keras.layers.ReLU()

    conv5_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm5_5 = tf.keras.layers.BatchNormalization()
    relu5_5 = tf.keras.layers.ReLU()

    conv5_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm5_6 = tf.keras.layers.BatchNormalization()
    relu5_6 = tf.keras.layers.ReLU()

    conv5_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm5_7 = tf.keras.layers.BatchNormalization()
    relu5_7 = tf.keras.layers.ReLU()

    conv5_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm5_8 = tf.keras.layers.BatchNormalization()
    relu5_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv2_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu2_1 = tf.keras.layers.ReLU()

    conv2_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_2 = tf.keras.layers.BatchNormalization()
    relu2_2 = tf.keras.layers.ReLU()

    conv2_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_3 = tf.keras.layers.BatchNormalization()
    relu2_3 = tf.keras.layers.ReLU()

    conv2_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_4 = tf.keras.layers.BatchNormalization()
    relu2_4 = tf.keras.layers.ReLU()

    conv2_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_5 = tf.keras.layers.BatchNormalization()
    relu2_5 = tf.keras.layers.ReLU()

    conv2_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_6 = tf.keras.layers.BatchNormalization()
    relu2_6 = tf.keras.layers.ReLU()

    conv2_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_7 = tf.keras.layers.BatchNormalization()
    relu2_7 = tf.keras.layers.ReLU()

    conv2_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_8 = tf.keras.layers.BatchNormalization()
    relu2_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv6_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu6_1 = tf.keras.layers.ReLU()

    conv6_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm6_2 = tf.keras.layers.BatchNormalization()
    relu6_2 = tf.keras.layers.ReLU()

    conv6_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm6_3 = tf.keras.layers.BatchNormalization()
    relu6_3 = tf.keras.layers.ReLU()

    conv6_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm6_4 = tf.keras.layers.BatchNormalization()
    relu6_4 = tf.keras.layers.ReLU()

    conv6_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm6_5 = tf.keras.layers.BatchNormalization()
    relu6_5 = tf.keras.layers.ReLU()

    conv6_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm6_6 = tf.keras.layers.BatchNormalization()
    relu6_6 = tf.keras.layers.ReLU()

    conv6_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm6_7 = tf.keras.layers.BatchNormalization()
    relu6_7 = tf.keras.layers.ReLU()

    conv6_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm6_8 = tf.keras.layers.BatchNormalization()
    relu6_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv_trans3_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_1 = tf.keras.layers.Dropout(0.5)
    batchnorm3_1 = tf.keras.layers.BatchNormalization()
    relu3_1 = tf.keras.layers.ReLU()
    cat3_1 = tf.keras.layers.Concatenate()

    conv_trans3_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_2 = tf.keras.layers.Dropout(0.5)
    batchnorm3_2 = tf.keras.layers.BatchNormalization()
    relu3_2 = tf.keras.layers.ReLU()
    cat3_2 = tf.keras.layers.Concatenate()

    conv_trans3_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_3 = tf.keras.layers.Dropout(0.5)
    batchnorm3_3 = tf.keras.layers.BatchNormalization()
    relu3_3 = tf.keras.layers.ReLU()
    cat3_3 = tf.keras.layers.Concatenate()

    conv_trans3_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm3_4 = tf.keras.layers.BatchNormalization()
    relu3_4 = tf.keras.layers.ReLU()
    cat3_4 = tf.keras.layers.Concatenate()

    conv_trans3_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm3_5 = tf.keras.layers.BatchNormalization()
    relu3_5 = tf.keras.layers.ReLU()
    cat3_5 = tf.keras.layers.Concatenate()

    conv_trans3_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm3_6 = tf.keras.layers.BatchNormalization()
    relu3_6 = tf.keras.layers.ReLU()
    cat3_6 = tf.keras.layers.Concatenate()

    conv_trans3_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm3_7 = tf.keras.layers.BatchNormalization()
    relu3_7 = tf.keras.layers.ReLU()
    cat3_7 = tf.keras.layers.Concatenate()

    conv_trans3_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

    # --------------------------------------------------------------------------------------------------------------
    conv_trans4_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_1 = tf.keras.layers.Dropout(0.5)
    batchnorm4_1 = tf.keras.layers.BatchNormalization()
    relu4_1 = tf.keras.layers.ReLU()
    cat4_1 = tf.keras.layers.Concatenate()

    conv_trans4_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_2 = tf.keras.layers.Dropout(0.5)
    batchnorm4_2 = tf.keras.layers.BatchNormalization()
    relu4_2 = tf.keras.layers.ReLU()
    cat4_2 = tf.keras.layers.Concatenate()

    conv_trans4_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop4_3 = tf.keras.layers.Dropout(0.5)
    batchnorm4_3 = tf.keras.layers.BatchNormalization()
    relu4_3 = tf.keras.layers.ReLU()
    cat4_3 = tf.keras.layers.Concatenate()

    conv_trans4_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm4_4 = tf.keras.layers.BatchNormalization()
    relu4_4 = tf.keras.layers.ReLU()
    cat4_4 = tf.keras.layers.Concatenate()

    conv_trans4_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm4_5 = tf.keras.layers.BatchNormalization()
    relu4_5 = tf.keras.layers.ReLU()
    cat4_5 = tf.keras.layers.Concatenate()

    conv_trans4_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm4_6 = tf.keras.layers.BatchNormalization()
    relu4_6 = tf.keras.layers.ReLU()
    cat4_6 = tf.keras.layers.Concatenate()

    conv_trans4_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm4_7 = tf.keras.layers.BatchNormalization()
    relu4_7 = tf.keras.layers.ReLU()
    cat4_7 = tf.keras.layers.Concatenate()

    conv_trans4_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

    # -------------------------------------------------------------------------------
    def encoder1(inputs):
        skips = []
        x = conv1_1(inputs)
        x = relu1_1(x)
        skips.append(x)

        x = conv1_2(x)
        x = batchnorm1_2(x)
        x = relu1_2(x)
        skips.append(x)

        x = conv1_3(x)
        x = batchnorm1_3(x)
        x = relu1_3(x)
        skips.append(x)

        x = conv1_4(x)
        x = batchnorm1_4(x)
        x = relu1_4(x)
        skips.append(x)

        x = conv1_5(x)
        x = batchnorm1_5(x)
        x = relu1_5(x)
        skips.append(x)

        x = conv1_6(x)
        x = batchnorm1_6(x)
        x = relu1_6(x)
        skips.append(x)

        x = conv1_7(x)
        x = batchnorm1_7(x)
        x = relu1_7(x)
        skips.append(x)

        x = conv1_8(x)
        x = batchnorm1_8(x)
        x = relu1_8(x)

        return x, skips

    def encoder2(inputs):
        skips = []
        x = conv2_1(inputs)
        x = relu2_1(x)
        skips.append(x)

        x = conv2_2(x)
        x = batchnorm2_2(x)
        x = relu2_2(x)
        skips.append(x)

        x = conv2_3(x)
        x = batchnorm2_3(x)
        x = relu2_3(x)
        skips.append(x)

        x = conv2_4(x)
        x = batchnorm2_4(x)
        x = relu2_4(x)
        skips.append(x)

        x = conv2_5(x)
        x = batchnorm2_5(x)
        x = relu2_5(x)
        skips.append(x)

        x = conv2_6(x)
        x = batchnorm2_6(x)
        x = relu2_6(x)
        skips.append(x)

        x = conv2_7(x)
        x = batchnorm2_7(x)
        x = relu2_7(x)
        skips.append(x)

        x = conv2_8(x)
        x = batchnorm2_8(x)
        x = relu2_8(x)

        return x, skips

    def encoder3(inputs):
        skips = []
        x = conv5_1(inputs)
        x = relu5_1(x)
        skips.append(x)

        x = conv5_2(x)
        x = batchnorm5_2(x)
        x = relu5_2(x)
        skips.append(x)

        x = conv5_3(x)
        x = batchnorm5_3(x)
        x = relu5_3(x)
        skips.append(x)

        x = conv5_4(x)
        x = batchnorm5_4(x)
        x = relu5_4(x)
        skips.append(x)

        x = conv5_5(x)
        x = batchnorm5_5(x)
        x = relu5_5(x)
        skips.append(x)

        x = conv5_6(x)
        x = batchnorm5_6(x)
        x = relu5_6(x)
        skips.append(x)

        x = conv5_7(x)
        x = batchnorm5_7(x)
        x = relu5_7(x)
        skips.append(x)

        x = conv5_8(x)
        x = batchnorm5_8(x)
        x = relu5_8(x)

        return x, skips

    def encoder4(inputs):
        skips = []
        x = conv6_1(inputs)
        x = relu6_1(x)
        skips.append(x)

        x = conv6_2(x)
        x = batchnorm6_2(x)
        x = relu6_2(x)
        skips.append(x)

        x = conv6_3(x)
        x = batchnorm6_3(x)
        x = relu6_3(x)
        skips.append(x)

        x = conv6_4(x)
        x = batchnorm6_4(x)
        x = relu6_4(x)
        skips.append(x)

        x = conv6_5(x)
        x = batchnorm6_5(x)
        x = relu6_5(x)
        skips.append(x)

        x = conv6_6(x)
        x = batchnorm6_6(x)
        x = relu6_6(x)
        skips.append(x)

        x = conv6_7(x)
        x = batchnorm6_7(x)
        x = relu6_7(x)
        skips.append(x)

        x = conv6_8(x)
        x = batchnorm6_8(x)
        x = relu6_8(x)

        return x, skips

    def decoder1(inputs, skips1, skips3):
        x = conv_trans3_1(inputs)
        x = batchnorm3_1(x)
        x = drop3_1(x)
        x = relu3_1(x)
        x = cat3_1([x, skips1[6]])
        x = cat3_1([x, skips3[6]])

        x = conv_trans3_2(x)
        x = batchnorm3_2(x)
        x = drop3_2(x)
        x = relu3_2(x)
        x = cat3_2([x, skips1[5]])
        x = cat3_2([x, skips3[5]])

        x = conv_trans3_3(x)
        x = batchnorm3_3(x)
        x = drop3_3(x)
        x = relu3_3(x)
        x = cat3_3([x, skips1[4]])
        x = cat3_3([x, skips3[4]])

        x = conv_trans3_4(x)
        x = batchnorm3_4(x)
        x = relu3_4(x)
        x = cat3_4([x, skips1[3]])
        x = cat3_4([x, skips3[3]])

        x = conv_trans3_5(x)
        x = batchnorm3_5(x)
        x = relu3_5(x)
        x = cat3_5([x, skips1[2]])
        x = cat3_5([x, skips3[2]])

        x = conv_trans3_6(x)
        x = batchnorm3_6(x)
        x = relu3_6(x)
        x = cat3_6([x, skips1[1]])
        x = cat3_6([x, skips3[1]])

        x = conv_trans3_7(x)
        x = batchnorm3_7(x)
        x = relu3_7(x)
        x = cat3_7([x, skips1[0]])
        x = cat3_7([x, skips3[0]])

        x = conv_trans3_last(x)
        return x

    def decoder2(inputs, skips1, skips3):
        x = conv_trans4_1(inputs)
        x = batchnorm4_1(x)
        x = drop4_1(x)
        x = relu4_1(x)
        x = cat4_1([x, skips1[6]])
        x = cat4_1([x, skips3[6]])

        x = conv_trans4_2(x)
        x = batchnorm4_2(x)
        x = drop4_2(x)
        x = relu4_2(x)
        x = cat4_2([x, skips1[5]])
        x = cat4_2([x, skips3[5]])

        x = conv_trans4_3(x)
        x = batchnorm4_3(x)
        x = drop4_3(x)
        x = relu4_3(x)
        x = cat4_3([x, skips1[4]])
        x = cat4_3([x, skips3[4]])

        x = conv_trans4_4(x)
        x = batchnorm4_4(x)
        x = relu4_4(x)
        x = cat4_4([x, skips1[3]])
        x = cat4_4([x, skips3[3]])

        x = conv_trans4_5(x)
        x = batchnorm4_5(x)
        x = relu4_5(x)
        x = cat4_5([x, skips1[2]])
        x = cat4_5([x, skips3[2]])

        x = conv_trans4_6(x)
        x = batchnorm4_6(x)
        x = relu4_6(x)
        x = cat4_6([x, skips1[1]])
        x = cat4_6([x, skips3[1]])

        x = conv_trans4_7(x)
        x = batchnorm4_7(x)
        x = relu4_7(x)
        x = cat4_7([x, skips1[0]])
        x = cat4_7([x, skips3[0]])

        x = conv_trans4_last(x)
        return x

    # Downsampling through the model
    spatial_out_list = []
    skips1_list = []
    skips3_list = []
    conved_output1_list = []

    # with tf.compat.v1.variable_scope('scope1', reuse=True):
    for t in range(0, inputs1.shape[5]):
        conved_output1, skips1_1 = encoder1((inputs1[:, :, :, :, :, t]))
        conved_output3, skips3_1 = encoder3((inputs2[:, :, :, :, :, t]))
        conved_output1 = tf.concat([conved_output1, conved_output3], axis=-1)
        skips1_list.append(skips1_1)
        skips3_list.append(skips3_1)
        conved_output1_list.append(conved_output1)


    # with tf.compat.v1.variable_scope('scope2', reuse=True):
    for conved_output1, skips1, skips3 in zip(conved_output1_list, skips1_list, skips3_list):
        spatial_out = decoder1(conved_output1, skips1, skips3)
        spatial_out_list.append(spatial_out)

    spatial_out_list = tf.convert_to_tensor(spatial_out_list)
    spatial_out_list = tf.transpose(spatial_out_list, [1, 2, 3, 4, 5, 0])

    inputs1_2 = tf.concat([tf.transpose(spatial_out_list, [0, 1, 2, 5, 4, 3]), tf.transpose(inputs1, [0, 1, 2, 5, 4, 3])],
                        axis=4)
    inputs2_2 = tf.concat(
        [tf.transpose(spatial_out_list, [0, 1, 2, 5, 4, 3]), tf.transpose(inputs2, [0, 1, 2, 5, 4, 3])],
        axis=4)

    skips2_list = []
    skips4_list = []
    conved_output2_list = []
    temporal_out_list = []
    # with tf.compat.v1.variable_scope('scope3', reuse=True):
    for z in range(0, inputs2_2.shape[5]):
        conved_output2, skips2 = encoder2((inputs1_2[:, :, :, :, :, z]))
        conved_output4, skips4 = encoder4((inputs2_2[:, :, :, :, :, z]))
        conved_output2 = tf.concat([conved_output2, conved_output4], axis=-1)
        skips2_list.append(skips2)
        skips4_list.append(skips4)
        conved_output2_list.append(conved_output2)

    # with tf.compat.v1.variable_scope('scope4', reuse=True):
    for conved_output2, skips2, skips4 in zip(conved_output2_list, skips2_list, skips4_list):
        temporal_out = decoder2(conved_output2, skips2, skips4)
        temporal_out_list.append(temporal_out)

    temporal_out_list = tf.convert_to_tensor(temporal_out_list)
    temporal_out_list = tf.transpose(temporal_out_list, [1, 2, 3, 0, 5, 4])


    return tf.keras.Model(inputs=[inputs1, inputs2], outputs=[spatial_out_list, temporal_out_list])

def Generator_correct_single_reuse_2input():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1, 10])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 16, 1, 10])

    initializer = tf.random_normal_initializer(0., 0.02)

    conv1_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu1_1 = tf.keras.layers.ReLU()

    conv1_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_2 = tf.keras.layers.BatchNormalization()
    relu1_2 = tf.keras.layers.ReLU()

    conv1_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_3 = tf.keras.layers.BatchNormalization()
    relu1_3 = tf.keras.layers.ReLU()

    conv1_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_4 = tf.keras.layers.BatchNormalization()
    relu1_4 = tf.keras.layers.ReLU()

    conv1_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_5 = tf.keras.layers.BatchNormalization()
    relu1_5 = tf.keras.layers.ReLU()

    conv1_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_6 = tf.keras.layers.BatchNormalization()
    relu1_6 = tf.keras.layers.ReLU()

    conv1_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_7 = tf.keras.layers.BatchNormalization()
    relu1_7 = tf.keras.layers.ReLU()

    conv1_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_8 = tf.keras.layers.BatchNormalization()
    relu1_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    conv_trans3_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_1 = tf.keras.layers.Dropout(0.5)
    batchnorm3_1 = tf.keras.layers.BatchNormalization()
    relu3_1 = tf.keras.layers.ReLU()
    cat3_1 = tf.keras.layers.Concatenate()

    conv_trans3_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_2 = tf.keras.layers.Dropout(0.5)
    batchnorm3_2 = tf.keras.layers.BatchNormalization()
    relu3_2 = tf.keras.layers.ReLU()
    cat3_2 = tf.keras.layers.Concatenate()

    conv_trans3_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_3 = tf.keras.layers.Dropout(0.5)
    batchnorm3_3 = tf.keras.layers.BatchNormalization()
    relu3_3 = tf.keras.layers.ReLU()
    cat3_3 = tf.keras.layers.Concatenate()

    conv_trans3_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_4 = tf.keras.layers.Dropout(0.5)
    batchnorm3_4 = tf.keras.layers.BatchNormalization()
    relu3_4 = tf.keras.layers.ReLU()
    cat3_4 = tf.keras.layers.Concatenate()

    conv_trans3_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_5 = tf.keras.layers.Dropout(0.5)
    batchnorm3_5 = tf.keras.layers.BatchNormalization()
    relu3_5 = tf.keras.layers.ReLU()
    cat3_5 = tf.keras.layers.Concatenate()

    conv_trans3_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_6 = tf.keras.layers.Dropout(0.5)
    batchnorm3_6 = tf.keras.layers.BatchNormalization()
    relu3_6 = tf.keras.layers.ReLU()
    cat3_6 = tf.keras.layers.Concatenate()

    conv_trans3_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_7 = tf.keras.layers.Dropout(0.5)
    batchnorm3_7 = tf.keras.layers.BatchNormalization()
    relu3_7 = tf.keras.layers.ReLU()
    cat3_7 = tf.keras.layers.Concatenate()

    conv_trans3_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

    # --------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    def encoder1(inputs):
        skips = []
        x = conv1_1(inputs)
        x = relu1_1(x)
        skips.append(x)

        x = conv1_2(x)
        x = batchnorm1_2(x)
        x = relu1_2(x)
        skips.append(x)

        x = conv1_3(x)
        x = batchnorm1_3(x)
        x = relu1_3(x)
        skips.append(x)

        x = conv1_4(x)
        x = batchnorm1_4(x)
        x = relu1_4(x)
        skips.append(x)

        x = conv1_5(x)
        x = batchnorm1_5(x)
        x = relu1_5(x)
        skips.append(x)

        x = conv1_6(x)
        x = batchnorm1_6(x)
        x = relu1_6(x)
        skips.append(x)

        x = conv1_7(x)
        x = batchnorm1_7(x)
        x = relu1_7(x)
        skips.append(x)

        x = conv1_8(x)
        x = batchnorm1_8(x)
        x = relu1_8(x)

        return x, skips

    def decoder1(inputs, skips):
        x = conv_trans3_1(inputs)
        x = batchnorm3_1(x)
        x = drop3_1(x)
        x = relu3_1(x)
        x = cat3_1([x, skips[6]])

        x = conv_trans3_2(x)
        x = batchnorm3_2(x)
        x = drop3_2(x)
        x = relu3_2(x)
        x = cat3_2([x, skips[5]])

        x = conv_trans3_3(x)
        x = batchnorm3_3(x)
        x = drop3_3(x)
        x = relu3_3(x)
        x = cat3_3([x, skips[4]])

        x = conv_trans3_4(x)
        x = batchnorm3_4(x)
        x = relu3_4(x)
        x = cat3_4([x, skips[3]])

        x = conv_trans3_5(x)
        x = batchnorm3_5(x)
        x = relu3_5(x)
        x = cat3_5([x, skips[2]])

        x = conv_trans3_6(x)
        x = batchnorm3_6(x)
        x = relu3_6(x)
        x = cat3_6([x, skips[1]])

        x = conv_trans3_7(x)
        x = batchnorm3_7(x)
        x = relu3_7(x)
        x = cat3_7([x, skips[0]])

        x = conv_trans3_last(x)
        return x

    # Downsampling through the model
    spatial_out_list = []
    skips1_list = []
    conved_output1_list = []

    concat_inputs1 = tf.concat([inputs1, inputs2],
                        axis=4)
    # with tf.compat.v1.variable_scope('scope1', reuse=True):
    for t in range(0, concat_inputs1.shape[5]):
        conved_output1, skips1_1 = encoder1((concat_inputs1[:, :, :, :, :, t]))
        skips1_list.append(skips1_1)
        conved_output1_list.append(conved_output1)

    # with tf.compat.v1.variable_scope('scope2', reuse=True):
    for conved_output1, skips1 in zip(conved_output1_list, skips1_list):
        spatial_out = decoder1(conved_output1, skips1)
        spatial_out_list.append(spatial_out)

    spatial_out_list = tf.convert_to_tensor(spatial_out_list)
    spatial_out_list = tf.transpose(spatial_out_list, [1, 2, 3, 4, 5, 0])


    return tf.keras.Model(inputs=[inputs1, inputs2], outputs=spatial_out_list)

def Generator_correct_connect_fold(sample_z, sample_t):
    inputs1 = tf.keras.layers.Input(shape=[256, 256, sample_z, 1, sample_t])
    initializer = tf.random_normal_initializer(0., 0.02)

    conv1_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu1_1 = tf.keras.layers.ReLU()

    conv1_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_2 = tf.keras.layers.BatchNormalization()
    relu1_2 = tf.keras.layers.ReLU()

    conv1_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_3 = tf.keras.layers.BatchNormalization()
    relu1_3 = tf.keras.layers.ReLU()

    conv1_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_4 = tf.keras.layers.BatchNormalization()
    relu1_4 = tf.keras.layers.ReLU()

    conv1_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_5 = tf.keras.layers.BatchNormalization()
    relu1_5 = tf.keras.layers.ReLU()

    conv1_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_6 = tf.keras.layers.BatchNormalization()
    relu1_6 = tf.keras.layers.ReLU()

    conv1_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_7 = tf.keras.layers.BatchNormalization()
    relu1_7 = tf.keras.layers.ReLU()

    conv1_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_8 = tf.keras.layers.BatchNormalization()
    relu1_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv2_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu2_1 = tf.keras.layers.ReLU()

    conv2_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_2 = tf.keras.layers.BatchNormalization()
    relu2_2 = tf.keras.layers.ReLU()

    conv2_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_3 = tf.keras.layers.BatchNormalization()
    relu2_3 = tf.keras.layers.ReLU()

    conv2_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_4 = tf.keras.layers.BatchNormalization()
    relu2_4 = tf.keras.layers.ReLU()

    conv2_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_5 = tf.keras.layers.BatchNormalization()
    relu2_5 = tf.keras.layers.ReLU()

    conv2_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_6 = tf.keras.layers.BatchNormalization()
    relu2_6 = tf.keras.layers.ReLU()

    conv2_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_7 = tf.keras.layers.BatchNormalization()
    relu2_7 = tf.keras.layers.ReLU()

    conv2_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_8 = tf.keras.layers.BatchNormalization()
    relu2_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    cat3_0 = tf.keras.layers.Concatenate()

    conv_trans3_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_1 = tf.keras.layers.Dropout(0.5)
    batchnorm3_1 = tf.keras.layers.BatchNormalization()
    relu3_1 = tf.keras.layers.ReLU()
    cat3_1 = tf.keras.layers.Concatenate()

    conv_trans3_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_2 = tf.keras.layers.Dropout(0.5)
    batchnorm3_2 = tf.keras.layers.BatchNormalization()
    relu3_2 = tf.keras.layers.ReLU()
    cat3_2 = tf.keras.layers.Concatenate()

    conv_trans3_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_3 = tf.keras.layers.Dropout(0.5)
    batchnorm3_3 = tf.keras.layers.BatchNormalization()
    relu3_3 = tf.keras.layers.ReLU()
    cat3_3 = tf.keras.layers.Concatenate()

    conv_trans3_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_4 = tf.keras.layers.Dropout(0.2)
    batchnorm3_4 = tf.keras.layers.BatchNormalization()
    relu3_4 = tf.keras.layers.ReLU()
    cat3_4 = tf.keras.layers.Concatenate()

    conv_trans3_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_5 = tf.keras.layers.Dropout(0.2)
    batchnorm3_5 = tf.keras.layers.BatchNormalization()
    relu3_5 = tf.keras.layers.ReLU()
    cat3_5 = tf.keras.layers.Concatenate()

    conv_trans3_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_6 = tf.keras.layers.Dropout(0.2)
    batchnorm3_6 = tf.keras.layers.BatchNormalization()
    relu3_6 = tf.keras.layers.ReLU()
    cat3_6 = tf.keras.layers.Concatenate()

    conv_trans3_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_7 = tf.keras.layers.Dropout(0.2)
    batchnorm3_7 = tf.keras.layers.BatchNormalization()
    relu3_7 = tf.keras.layers.ReLU()
    cat3_7 = tf.keras.layers.Concatenate()

    drop3_8 = tf.keras.layers.Dropout(0.2)

    conv_trans3_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

    # --------------------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------

    def encoder1(inputs):
        skips = []
        x = conv1_1(inputs)
        x = relu1_1(x)
        skips.append(x)

        x = conv1_2(x)
        x = batchnorm1_2(x)
        x = relu1_2(x)
        skips.append(x)

        x = conv1_3(x)
        x = batchnorm1_3(x)
        x = relu1_3(x)
        skips.append(x)

        x = conv1_4(x)
        x = batchnorm1_4(x)
        x = relu1_4(x)
        skips.append(x)

        x = conv1_5(x)
        x = batchnorm1_5(x)
        x = relu1_5(x)
        skips.append(x)

        x = conv1_6(x)
        x = batchnorm1_6(x)
        x = relu1_6(x)
        skips.append(x)

        x = conv1_7(x)
        x = batchnorm1_7(x)
        x = relu1_7(x)
        skips.append(x)

        x = conv1_8(x)
        x = batchnorm1_8(x)
        x = relu1_8(x)

        return x, skips

    def encoder2(inputs):
        skips = []
        x = conv2_1(inputs)
        x = relu2_1(x)
        skips.append(x)

        x = conv2_2(x)
        x = batchnorm2_2(x)
        x = relu2_2(x)
        skips.append(x)

        x = conv2_3(x)
        x = batchnorm2_3(x)
        x = relu2_3(x)
        skips.append(x)

        x = conv2_4(x)
        x = batchnorm2_4(x)
        x = relu2_4(x)
        skips.append(x)

        x = conv2_5(x)
        x = batchnorm2_5(x)
        x = relu2_5(x)
        skips.append(x)

        x = conv2_6(x)
        x = batchnorm2_6(x)
        x = relu2_6(x)
        skips.append(x)

        x = conv2_7(x)
        x = batchnorm2_7(x)
        x = relu2_7(x)
        skips.append(x)

        x = conv2_8(x)
        x = batchnorm2_8(x)
        x = relu2_8(x)

        return x, skips

    def decoder1(inputs1, inputs2, skips1, skips2):
        x = cat3_0([inputs1, inputs2])

        x = conv_trans3_1(x)
        x = batchnorm3_1(x)
        x = drop3_1(x)
        x = relu3_1(x)
        x = cat3_1([x, skips1[6], skips2[6]])

        x = conv_trans3_2(x)
        x = batchnorm3_2(x)
        x = drop3_2(x)
        x = relu3_2(x)
        x = cat3_2([x, skips1[5], skips2[5]])

        x = conv_trans3_3(x)
        x = batchnorm3_3(x)
        x = drop3_3(x)
        x = relu3_3(x)
        x = cat3_3([x, skips1[4], skips2[4]])

        x = conv_trans3_4(x)
        x = batchnorm3_4(x)
        # x = drop3_4(x)
        x = relu3_4(x)
        x = cat3_4([x, skips1[3], skips2[3]])

        x = conv_trans3_5(x)
        x = batchnorm3_5(x)
        # x = drop3_5(x)
        x = relu3_5(x)
        x = cat3_5([x, skips1[2], skips2[2]])

        x = conv_trans3_6(x)
        x = batchnorm3_6(x)
        # x = drop3_6(x)
        x = relu3_6(x)
        x = cat3_6([x, skips1[1], skips2[1]])

        x = conv_trans3_7(x)
        x = batchnorm3_7(x)
        # x = drop3_7(x)
        x = relu3_7(x)
        x = cat3_7([x, skips1[0], skips2[0]])

        # x = drop3_7(x)
        x = conv_trans3_last(x)
        return x


    # Downsampling through the model
    spatial_out_list = []
    skips1_list = []
    conved_output1_list = []

    # with tf.compat.v1.variable_scope('scope1', reuse=True):
    for t in range(0, inputs1.shape[5]):
        conved_output1, skips1_1 = encoder1((inputs1[:, :, :, :, :, t]))
        skips1_list.append(skips1_1)
        conved_output1_list.append(conved_output1)

    inputs2 = tf.transpose(inputs1, [0, 1, 2, 5, 4, 3])

    skips2_list = []
    conved_output2_list = []
    # with tf.compat.v1.variable_scope('scope3', reuse=True):
    for z in range(0, inputs2.shape[5]):
        conved_output2, skips2 = encoder2((inputs2[:, :, :, :, :, z]))
        skips2_list.append(skips2)
        conved_output2_list.append(conved_output2)

    new_conv_list = []
    new_skip_list = []
    for t in range(0,sample_t):
        new_conv_list.append([])
        new_skip_list.append([])
        for l in range(0, 7):
            new_skip_list[t].append([])
        for z in range(0,sample_z):
            # print(conved_output2_list[z].shape)
            # print(skips2_list[z][0].shape)
            new_conv_list[t].append(conved_output2_list[z][:,:,:,t,:])
            for l in range(0,7):
                new_skip_list[t][l].append(skips2_list[z][l][:,:,:,t,:])

    new_conv_tensor = []
    new_skip_tensor = []
    for t in range(0, sample_t):
        # print(tf.convert_to_tensor(new_conv_list[t]).shape)
        temp = tf.convert_to_tensor(new_conv_list[t])
        temp = tf.transpose(temp, [1,2,3,0,4])
        new_conv_tensor.append(temp)
        new_skip_tensor.append([])
        for l in range(0, 7):
            # print(tf.convert_to_tensor(new_skip_list[t][l]).shape)
            temp = tf.convert_to_tensor(new_skip_list[t][l])
            temp = tf.transpose(temp, [1, 2, 3, 0, 4])
            new_skip_tensor[t].append(temp)



    # with tf.compat.v1.variable_scope('scope2', reuse=True):
    for conved_output1, conved_output2, skips1, skips2 in zip(conved_output1_list, new_conv_tensor, skips1_list, new_skip_tensor):
        spatial_out = decoder1(conved_output1,conved_output2, skips1, skips2)
        spatial_out_list.append(spatial_out)

    spatial_out_list = tf.convert_to_tensor(spatial_out_list)
    spatial_out_list = tf.transpose(spatial_out_list, [1, 2, 3, 4, 5, 0])

    del skips1_list, conved_output1_list, skips2_list, conved_output2_list, new_conv_list, new_skip_list, new_conv_tensor, new_skip_tensor, inputs2
    gc.collect()

    return tf.keras.Model(inputs=[inputs1], outputs=spatial_out_list)

def Generator_correct_connect_fold_guide(sample_z, sample_t):
    inputs1 = tf.keras.layers.Input(shape=[256, 256, sample_z, 1, sample_t])
    initializer = tf.random_normal_initializer(0., 0.02)

    conv1_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu1_1 = tf.keras.layers.ReLU()

    conv1_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_2 = tf.keras.layers.BatchNormalization()
    relu1_2 = tf.keras.layers.ReLU()

    conv1_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_3 = tf.keras.layers.BatchNormalization()
    relu1_3 = tf.keras.layers.ReLU()

    conv1_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_4 = tf.keras.layers.BatchNormalization()
    relu1_4 = tf.keras.layers.ReLU()

    conv1_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_5 = tf.keras.layers.BatchNormalization()
    relu1_5 = tf.keras.layers.ReLU()

    conv1_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_6 = tf.keras.layers.BatchNormalization()
    relu1_6 = tf.keras.layers.ReLU()

    conv1_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_7 = tf.keras.layers.BatchNormalization()
    relu1_7 = tf.keras.layers.ReLU()

    conv1_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_8 = tf.keras.layers.BatchNormalization()
    relu1_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv2_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu2_1 = tf.keras.layers.ReLU()

    conv2_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_2 = tf.keras.layers.BatchNormalization()
    relu2_2 = tf.keras.layers.ReLU()

    conv2_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_3 = tf.keras.layers.BatchNormalization()
    relu2_3 = tf.keras.layers.ReLU()

    conv2_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_4 = tf.keras.layers.BatchNormalization()
    relu2_4 = tf.keras.layers.ReLU()

    conv2_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_5 = tf.keras.layers.BatchNormalization()
    relu2_5 = tf.keras.layers.ReLU()

    conv2_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_6 = tf.keras.layers.BatchNormalization()
    relu2_6 = tf.keras.layers.ReLU()

    conv2_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_7 = tf.keras.layers.BatchNormalization()
    relu2_7 = tf.keras.layers.ReLU()

    conv2_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_8 = tf.keras.layers.BatchNormalization()
    relu2_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    cat3_0 = tf.keras.layers.Concatenate()

    conv_trans3_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_1 = tf.keras.layers.Dropout(0.5)
    batchnorm3_1 = tf.keras.layers.BatchNormalization()
    relu3_1 = tf.keras.layers.ReLU()
    cat3_1 = tf.keras.layers.Concatenate()

    conv_trans3_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_2 = tf.keras.layers.Dropout(0.5)
    batchnorm3_2 = tf.keras.layers.BatchNormalization()
    relu3_2 = tf.keras.layers.ReLU()
    cat3_2 = tf.keras.layers.Concatenate()

    conv_trans3_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_3 = tf.keras.layers.Dropout(0.5)
    batchnorm3_3 = tf.keras.layers.BatchNormalization()
    relu3_3 = tf.keras.layers.ReLU()
    cat3_3 = tf.keras.layers.Concatenate()

    conv_trans3_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_4 = tf.keras.layers.Dropout(0.2)
    batchnorm3_4 = tf.keras.layers.BatchNormalization()
    relu3_4 = tf.keras.layers.ReLU()
    cat3_4 = tf.keras.layers.Concatenate()

    conv_trans3_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_5 = tf.keras.layers.Dropout(0.2)
    batchnorm3_5 = tf.keras.layers.BatchNormalization()
    relu3_5 = tf.keras.layers.ReLU()
    cat3_5 = tf.keras.layers.Concatenate()

    conv_trans3_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_6 = tf.keras.layers.Dropout(0.2)
    batchnorm3_6 = tf.keras.layers.BatchNormalization()
    relu3_6 = tf.keras.layers.ReLU()
    cat3_6 = tf.keras.layers.Concatenate()

    conv_trans3_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_7 = tf.keras.layers.Dropout(0.2)
    batchnorm3_7 = tf.keras.layers.BatchNormalization()
    relu3_7 = tf.keras.layers.ReLU()
    cat3_7 = tf.keras.layers.Concatenate()

    conv_trans3_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

    # --------------------------------------------------------------------------------------------------------------

    conv_spade_1_1 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                       strides=1,
                                                       padding='same',
                                                       kernel_initializer=initializer)
    relu_spade_1_1 = tf.keras.layers.ReLU()
    conv_spade_1_2 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    conv_spade_1_3 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    mul_spade_1_1 = tf.keras.layers.Multiply()
    add_spade_1_1 = tf.keras.layers.Add()
    relu_spade_1_2 = tf.keras.layers.ReLU()
    conv_spade_1_4 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    add_spade_1_2 = tf.keras.layers.Add()

    #****************************
    conv_spade_2_1 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    relu_spade_2_1 = tf.keras.layers.ReLU()
    conv_spade_2_2 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    conv_spade_2_3 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    mul_spade_2_1 = tf.keras.layers.Multiply()
    add_spade_2_1 = tf.keras.layers.Add()
    relu_spade_2_2 = tf.keras.layers.ReLU()
    conv_spade_2_4 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    add_spade_2_2 = tf.keras.layers.Add()

    # ****************************
    conv_spade_3_1 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    relu_spade_3_1 = tf.keras.layers.ReLU()
    conv_spade_3_2 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    conv_spade_3_3 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    mul_spade_3_1 = tf.keras.layers.Multiply()
    add_spade_3_1 = tf.keras.layers.Add()
    relu_spade_3_2 = tf.keras.layers.ReLU()
    conv_spade_3_4 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    add_spade_3_2 = tf.keras.layers.Add()

    # ****************************
    conv_spade_4_1 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    relu_spade_4_1 = tf.keras.layers.ReLU()
    conv_spade_4_2 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    conv_spade_4_3 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    mul_spade_4_1 = tf.keras.layers.Multiply()
    add_spade_4_1 = tf.keras.layers.Add()
    relu_spade_4_2 = tf.keras.layers.ReLU()
    conv_spade_4_4 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    add_spade_4_2 = tf.keras.layers.Add()


    #****************************
    conv_spade_5_1 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    relu_spade_5_1 = tf.keras.layers.ReLU()
    conv_spade_5_2 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    conv_spade_5_3 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    mul_spade_5_1 = tf.keras.layers.Multiply()
    add_spade_5_1 = tf.keras.layers.Add()
    relu_spade_5_2 = tf.keras.layers.ReLU()
    conv_spade_5_4 = tf.keras.layers.Conv3DTranspose(256, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    add_spade_5_2 = tf.keras.layers.Add()

    # ****************************
    conv_spade_6_1 = tf.keras.layers.Conv3DTranspose(128, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    relu_spade_6_1 = tf.keras.layers.ReLU()
    conv_spade_6_2 = tf.keras.layers.Conv3DTranspose(128, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    conv_spade_6_3 = tf.keras.layers.Conv3DTranspose(128, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    mul_spade_6_1 = tf.keras.layers.Multiply()
    add_spade_6_1 = tf.keras.layers.Add()
    relu_spade_6_2 = tf.keras.layers.ReLU()
    conv_spade_6_4 = tf.keras.layers.Conv3DTranspose(128, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    add_spade_6_2 = tf.keras.layers.Add()

    # ****************************
    conv_spade_7_1 = tf.keras.layers.Conv3DTranspose(64, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    relu_spade_7_1 = tf.keras.layers.ReLU()
    conv_spade_7_2 = tf.keras.layers.Conv3DTranspose(64, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    conv_spade_7_3 = tf.keras.layers.Conv3DTranspose(64, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    mul_spade_7_1 = tf.keras.layers.Multiply()
    add_spade_7_1 = tf.keras.layers.Add()
    relu_spade_7_2 = tf.keras.layers.ReLU()
    conv_spade_7_4 = tf.keras.layers.Conv3DTranspose(64, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    add_spade_7_2 = tf.keras.layers.Add()

    # ****************************
    conv_spade_8_1 = tf.keras.layers.Conv3DTranspose(32, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    relu_spade_8_1 = tf.keras.layers.ReLU()
    conv_spade_8_2 = tf.keras.layers.Conv3DTranspose(32, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    conv_spade_8_3 = tf.keras.layers.Conv3DTranspose(32, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    mul_spade_8_1 = tf.keras.layers.Multiply()
    add_spade_8_1 = tf.keras.layers.Add()
    relu_spade_8_2 = tf.keras.layers.ReLU()
    conv_spade_8_4 = tf.keras.layers.Conv3DTranspose(32, 4,
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=initializer)
    add_spade_8_2 = tf.keras.layers.Add()




    #------------------------------------------------------------------------------------------------------------------
    def spade_norm_1(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = conv_spade_1_1(inputs2)
        x2 = relu_spade_1_1(x2)
        x2_m = conv_spade_1_2(x2)
        x2_a = conv_spade_1_3(x2)
        x = mul_spade_1_1([inputs1, x2_m])
        x = add_spade_1_1([x, x2_a])
        x = relu_spade_1_2(x)
        x = conv_spade_1_4(x)
        x = add_spade_1_2([x, inputs1])
        return x
    def spade_norm_2(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = conv_spade_2_1(inputs2)
        x2 = relu_spade_2_1(x2)
        x2_m = conv_spade_2_2(x2)
        x2_a = conv_spade_2_3(x2)
        x = mul_spade_2_1([inputs1, x2_m])
        x = add_spade_2_1([x, x2_a])
        x = relu_spade_2_2(x)
        x = conv_spade_2_4(x)
        x = add_spade_2_2([x, inputs1])
        return x
    def spade_norm_3(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = conv_spade_3_1(inputs2)
        x2 = relu_spade_3_1(x2)
        x2_m = conv_spade_3_2(x2)
        x2_a = conv_spade_3_3(x2)
        x = mul_spade_3_1([inputs1, x2_m])
        x = add_spade_3_1([x, x2_a])
        x = relu_spade_3_2(x)
        x = conv_spade_3_4(x)
        x = add_spade_3_2([x, inputs1])
        return x
    def spade_norm_4(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = conv_spade_4_1(inputs2)
        x2 = relu_spade_4_1(x2)
        x2_m = conv_spade_4_2(x2)
        x2_a = conv_spade_4_3(x2)
        x = mul_spade_4_1([inputs1, x2_m])
        x = add_spade_4_1([x, x2_a])
        x = relu_spade_4_2(x)
        x = conv_spade_4_4(x)
        x = add_spade_4_2([x, inputs1])
        return x
    def spade_norm_5(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = conv_spade_5_1(inputs2)
        x2 = relu_spade_5_1(x2)
        x2_m = conv_spade_5_2(x2)
        x2_a = conv_spade_5_3(x2)
        x = mul_spade_5_1([inputs1, x2_m])
        x = add_spade_5_1([x, x2_a])
        x = relu_spade_5_2(x)
        x = conv_spade_5_4(x)
        x = add_spade_5_2([x, inputs1])
        return x
    def spade_norm_6(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = conv_spade_6_1(inputs2)
        x2 = relu_spade_6_1(x2)
        x2_m = conv_spade_6_2(x2)
        x2_a = conv_spade_6_3(x2)
        x = mul_spade_6_1([inputs1, x2_m])
        x = add_spade_6_1([x, x2_a])
        x = relu_spade_6_2(x)
        x = conv_spade_6_4(x)
        x = add_spade_6_2([x, inputs1])
        return x
    def spade_norm_7(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = conv_spade_7_1(inputs2)
        x2 = relu_spade_7_1(x2)
        x2_m = conv_spade_7_2(x2)
        x2_a = conv_spade_7_3(x2)
        x = mul_spade_7_1([inputs1, x2_m])
        x = add_spade_7_1([x, x2_a])
        x = relu_spade_7_2(x)
        x = conv_spade_7_4(x)
        x = add_spade_7_2([x, inputs1])
        return x
    def spade_norm_8(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = conv_spade_8_1(inputs2)
        x2 = relu_spade_8_1(x2)
        x2_m = conv_spade_8_2(x2)
        x2_a = conv_spade_8_3(x2)
        x = mul_spade_8_1([inputs1, x2_m])
        x = add_spade_8_1([x, x2_a])
        x = relu_spade_8_2(x)
        x = conv_spade_8_4(x)
        x = add_spade_8_2([x, inputs1])
        return x
    #------------------------------------------------------------------------------------------

    def encoder1(inputs):
        skips = []
        x = conv1_1(inputs)
        x = relu1_1(x)
        skips.append(x)

        x = conv1_2(x)
        x = batchnorm1_2(x)
        x = relu1_2(x)
        skips.append(x)

        x = conv1_3(x)
        x = batchnorm1_3(x)
        x = relu1_3(x)
        skips.append(x)

        x = conv1_4(x)
        x = batchnorm1_4(x)
        x = relu1_4(x)
        skips.append(x)

        x = conv1_5(x)
        x = batchnorm1_5(x)
        x = relu1_5(x)
        skips.append(x)

        x = conv1_6(x)
        x = batchnorm1_6(x)
        x = relu1_6(x)
        skips.append(x)

        x = conv1_7(x)
        x = batchnorm1_7(x)
        x = relu1_7(x)
        skips.append(x)

        x = conv1_8(x)
        x = batchnorm1_8(x)
        x = relu1_8(x)

        return x, skips

    def encoder2(inputs):
        skips = []
        x = conv2_1(inputs)
        x = relu2_1(x)
        skips.append(x)

        x = conv2_2(x)
        x = batchnorm2_2(x)
        x = relu2_2(x)
        skips.append(x)

        x = conv2_3(x)
        x = batchnorm2_3(x)
        x = relu2_3(x)
        skips.append(x)

        x = conv2_4(x)
        x = batchnorm2_4(x)
        x = relu2_4(x)
        skips.append(x)

        x = conv2_5(x)
        x = batchnorm2_5(x)
        x = relu2_5(x)
        skips.append(x)

        x = conv2_6(x)
        x = batchnorm2_6(x)
        x = relu2_6(x)
        skips.append(x)

        x = conv2_7(x)
        x = batchnorm2_7(x)
        x = relu2_7(x)
        skips.append(x)

        x = conv2_8(x)
        x = batchnorm2_8(x)
        x = relu2_8(x)

        return x, skips

    def decoder1(inputs1, inputs2, skips1, skips2):
        # x = cat3_0([inputs1, inputs2])
        x = spade_norm_1(inputs1, inputs2, 256, 4)

        x = conv_trans3_1(x)
        x = batchnorm3_1(x)
        x = drop3_1(x)
        x = relu3_1(x)
        x = spade_norm_2(x, skips2[6], 256, 4)
        x = cat3_1([x, skips1[6]])


        x = conv_trans3_2(x)
        x = batchnorm3_2(x)
        x = drop3_2(x)
        x = relu3_2(x)
        x = spade_norm_3(x, skips2[5], 256, 4)
        x = cat3_2([x, skips1[5]])


        x = conv_trans3_3(x)
        x = batchnorm3_3(x)
        x = drop3_3(x)
        x = relu3_3(x)
        x = spade_norm_4(x, skips2[4], 256, 4)
        x = cat3_3([x, skips1[4]])


        x = conv_trans3_4(x)
        x = batchnorm3_4(x)
        # x = drop3_4(x)
        x = relu3_4(x)
        x = spade_norm_5(x, skips2[3], 256, 4)
        # x = drop3_4(x)
        x = cat3_4([x, skips1[3]])


        x = conv_trans3_5(x)
        x = batchnorm3_5(x)
        # x = drop3_5(x)
        x = relu3_5(x)
        x = spade_norm_6(x, skips2[2], 128, 4)
        # x = drop3_5(x)
        x = cat3_5([x, skips1[2]])


        x = conv_trans3_6(x)
        x = batchnorm3_6(x)
        # x = drop3_6(x)
        x = relu3_6(x)
        x = spade_norm_7(x, skips2[1], 64, 4)
        # x = drop3_6(x)
        x = cat3_6([x, skips1[1]])


        x = conv_trans3_7(x)
        x = batchnorm3_7(x)
        # x = drop3_7(x)
        x = relu3_7(x)
        x = spade_norm_8(x, skips2[0], 32, 4)
        # x = drop3_7(x)
        x = cat3_7([x, skips1[0]])

        # x = drop3_7(x)
        x = conv_trans3_last(x)
        return x




    # Downsampling through the model
    spatial_out_list = []
    skips1_list = []
    conved_output1_list = []

    # with tf.compat.v1.variable_scope('scope1', reuse=True):
    for t in range(0, inputs1.shape[5]):
        conved_output1, skips1_1 = encoder1((inputs1[:, :, :, :, :, t]))
        skips1_list.append(skips1_1)
        conved_output1_list.append(conved_output1)

    inputs2 = tf.transpose(inputs1, [0, 1, 2, 5, 4, 3])

    skips2_list = []
    conved_output2_list = []
    # with tf.compat.v1.variable_scope('scope3', reuse=True):
    for z in range(0, inputs2.shape[5]):
        conved_output2, skips2 = encoder2((inputs2[:, :, :, :, :, z]))
        skips2_list.append(skips2)
        conved_output2_list.append(conved_output2)

    new_conv_list = []
    new_skip_list = []
    for t in range(0,sample_t):
        new_conv_list.append([])
        new_skip_list.append([])
        for l in range(0, 7):
            new_skip_list[t].append([])
        for z in range(0,sample_z):
            # print(conved_output2_list[z].shape)
            # print(skips2_list[z][0].shape)
            new_conv_list[t].append(conved_output2_list[z][:,:,:,t,:])
            for l in range(0,7):
                new_skip_list[t][l].append(skips2_list[z][l][:,:,:,t,:])

    new_conv_tensor = []
    new_skip_tensor = []
    for t in range(0, sample_t):
        # print(tf.convert_to_tensor(new_conv_list[t]).shape)
        temp = tf.convert_to_tensor(new_conv_list[t])
        temp = tf.transpose(temp, [1,2,3,0,4])
        new_conv_tensor.append(temp)
        new_skip_tensor.append([])
        for l in range(0, 7):
            # print(tf.convert_to_tensor(new_skip_list[t][l]).shape)
            temp = tf.convert_to_tensor(new_skip_list[t][l])
            temp = tf.transpose(temp, [1, 2, 3, 0, 4])
            new_skip_tensor[t].append(temp)



    # with tf.compat.v1.variable_scope('scope2', reuse=True):
    for conved_output1, conved_output2, skips1, skips2 in zip(conved_output1_list, new_conv_tensor, skips1_list, new_skip_tensor):
        spatial_out = decoder1(conved_output1,conved_output2, skips1, skips2)
        spatial_out_list.append(spatial_out)

    spatial_out_list = tf.convert_to_tensor(spatial_out_list)
    spatial_out_list = tf.transpose(spatial_out_list, [1, 2, 3, 4, 5, 0])

    del skips1_list, conved_output1_list, skips2_list, conved_output2_list, new_conv_list, new_skip_list, new_conv_tensor, new_skip_tensor, inputs2
    gc.collect()

    return tf.keras.Model(inputs=[inputs1], outputs=spatial_out_list)

def Generator_correct_connect_guide():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 16, 1, 10])
    initializer = tf.random_normal_initializer(0., 0.02)

    conv1_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu1_1 = tf.keras.layers.ReLU()

    conv1_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_2 = tf.keras.layers.BatchNormalization()
    relu1_2 = tf.keras.layers.ReLU()

    conv1_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_3 = tf.keras.layers.BatchNormalization()
    relu1_3 = tf.keras.layers.ReLU()

    conv1_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_4 = tf.keras.layers.BatchNormalization()
    relu1_4 = tf.keras.layers.ReLU()

    conv1_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_5 = tf.keras.layers.BatchNormalization()
    relu1_5 = tf.keras.layers.ReLU()

    conv1_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_6 = tf.keras.layers.BatchNormalization()
    relu1_6 = tf.keras.layers.ReLU()

    conv1_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_7 = tf.keras.layers.BatchNormalization()
    relu1_7 = tf.keras.layers.ReLU()

    conv1_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_8 = tf.keras.layers.BatchNormalization()
    relu1_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv2_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu2_1 = tf.keras.layers.ReLU()

    conv2_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_2 = tf.keras.layers.BatchNormalization()
    relu2_2 = tf.keras.layers.ReLU()

    conv2_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_3 = tf.keras.layers.BatchNormalization()
    relu2_3 = tf.keras.layers.ReLU()

    conv2_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_4 = tf.keras.layers.BatchNormalization()
    relu2_4 = tf.keras.layers.ReLU()

    conv2_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_5 = tf.keras.layers.BatchNormalization()
    relu2_5 = tf.keras.layers.ReLU()

    conv2_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_6 = tf.keras.layers.BatchNormalization()
    relu2_6 = tf.keras.layers.ReLU()

    conv2_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_7 = tf.keras.layers.BatchNormalization()
    relu2_7 = tf.keras.layers.ReLU()

    conv2_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm2_8 = tf.keras.layers.BatchNormalization()
    relu2_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv_trans3_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_1 = tf.keras.layers.Dropout(0.5)
    batchnorm3_1 = tf.keras.layers.BatchNormalization()
    relu3_1 = tf.keras.layers.ReLU()
    cat3_1 = tf.keras.layers.Concatenate()

    conv_trans3_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_2 = tf.keras.layers.Dropout(0.5)
    batchnorm3_2 = tf.keras.layers.BatchNormalization()
    relu3_2 = tf.keras.layers.ReLU()
    cat3_2 = tf.keras.layers.Concatenate()

    conv_trans3_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_3 = tf.keras.layers.Dropout(0.5)
    batchnorm3_3 = tf.keras.layers.BatchNormalization()
    relu3_3 = tf.keras.layers.ReLU()
    cat3_3 = tf.keras.layers.Concatenate()

    conv_trans3_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm3_4 = tf.keras.layers.BatchNormalization()
    relu3_4 = tf.keras.layers.ReLU()
    cat3_4 = tf.keras.layers.Concatenate()

    conv_trans3_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm3_5 = tf.keras.layers.BatchNormalization()
    relu3_5 = tf.keras.layers.ReLU()
    cat3_5 = tf.keras.layers.Concatenate()

    conv_trans3_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm3_6 = tf.keras.layers.BatchNormalization()
    relu3_6 = tf.keras.layers.ReLU()
    cat3_6 = tf.keras.layers.Concatenate()

    conv_trans3_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    batchnorm3_7 = tf.keras.layers.BatchNormalization()
    relu3_7 = tf.keras.layers.ReLU()
    cat3_7 = tf.keras.layers.Concatenate()

    conv_trans3_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

#-----------------------------------------------------------------------------------------

    def spade_norm(inputs1, inputs2, channel, size):
        # x1 = tf.keras.layers.BatchNormalization()(inputs1)
        x2 = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(inputs2)
        x2 = tf.keras.layers.ReLU()(x2)
        x2_m = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                   kernel_initializer=initializer, use_bias=False)(x2)
        x2_a = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False)(x2)
        x = tf.math.multiply(inputs1, x2_m)
        x = tf.math.add(x, x2_a)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(channel, size, strides=1, padding='same',
                                    kernel_initializer=initializer, use_bias=False)(x)
        x = tf.math.add(x, inputs1)
        return x

    def encoder1(inputs, skips1_2):
        skips = []
        x = conv1_1(inputs)
        x = relu1_1(x)
        x = spade_norm(x, skips1_2[0], 32, 4)
        skips.append(x)

        x = conv1_2(x)
        x = batchnorm1_2(x)
        x = relu1_2(x)
        x = spade_norm(x, skips1_2[1], 64, 4)
        skips.append(x)

        x = conv1_3(x)
        x = batchnorm1_3(x)
        x = relu1_3(x)
        x = spade_norm(x, skips1_2[2], 128, 4)
        skips.append(x)

        x = conv1_4(x)
        x = batchnorm1_4(x)
        x = relu1_4(x)
        x = spade_norm(x, skips1_2[3], 256, 4)
        skips.append(x)

        x = conv1_5(x)
        x = batchnorm1_5(x)
        x = relu1_5(x)
        x = spade_norm(x, skips1_2[4], 256, 4)
        skips.append(x)

        x = conv1_6(x)
        x = batchnorm1_6(x)
        x = relu1_6(x)
        x = spade_norm(x, skips1_2[5], 256, 4)
        skips.append(x)

        x = conv1_7(x)
        x = batchnorm1_7(x)
        x = relu1_7(x)
        x = spade_norm(x, skips1_2[6], 256, 4)
        skips.append(x)

        x = conv1_8(x)
        x = batchnorm1_8(x)
        x = relu1_8(x)

        return x, skips

    def encoder2(inputs):
        skips = []
        x = conv2_1(inputs)
        x = relu2_1(x)
        skips.append(x)

        x = conv2_2(x)
        x = batchnorm2_2(x)
        x = relu2_2(x)
        skips.append(x)

        x = conv2_3(x)
        x = batchnorm2_3(x)
        x = relu2_3(x)
        skips.append(x)

        x = conv2_4(x)
        x = batchnorm2_4(x)
        x = relu2_4(x)
        skips.append(x)

        x = conv2_5(x)
        x = batchnorm2_5(x)
        x = relu2_5(x)
        skips.append(x)

        x = conv2_6(x)
        x = batchnorm2_6(x)
        x = relu2_6(x)
        skips.append(x)

        x = conv2_7(x)
        x = batchnorm2_7(x)
        x = relu2_7(x)
        skips.append(x)

        x = conv2_8(x)
        x = batchnorm2_8(x)
        x = relu2_8(x)

        return x, skips

    def decoder1(inputs, skips):
        x = conv_trans3_1(inputs)
        x = batchnorm3_1(x)
        x = drop3_1(x)
        x = relu3_1(x)
        x = cat3_1([x, skips[6]])

        x = conv_trans3_2(x)
        x = batchnorm3_2(x)
        x = drop3_2(x)
        x = relu3_2(x)
        x = cat3_2([x, skips[5]])

        x = conv_trans3_3(x)
        x = batchnorm3_3(x)
        x = drop3_3(x)
        x = relu3_3(x)
        x = cat3_3([x, skips[4]])

        x = conv_trans3_4(x)
        x = batchnorm3_4(x)
        x = relu3_4(x)
        x = cat3_4([x, skips[3]])

        x = conv_trans3_5(x)
        x = batchnorm3_5(x)
        x = relu3_5(x)
        x = cat3_5([x, skips[2]])

        x = conv_trans3_6(x)
        x = batchnorm3_6(x)
        x = relu3_6(x)
        x = cat3_6([x, skips[1]])

        x = conv_trans3_7(x)
        x = batchnorm3_7(x)
        x = relu3_7(x)
        x = cat3_7([x, skips[0]])

        x = conv_trans3_last(x)
        return x


    # Downsampling through the model
    spatial_out_list = []

    # with tf.compat.v1.variable_scope('scope1', reuse=True):
    input_pre = tf.ones([1, 256, 256, 16, 1])
    for t in range(0, inputs1.shape[5]):
        conved_output2, skips1_2 = encoder2((input_pre))
        conved_output1, skips1_1 = encoder1((inputs1[:, :, :, :, :, t]),skips1_2)
        spatial_out = decoder1(conved_output1, skips1_1)
        spatial_out_list.append(spatial_out)
        input_pre = spatial_out


    spatial_out_list = tf.convert_to_tensor(spatial_out_list)
    spatial_out_list = tf.transpose(spatial_out_list, [1, 2, 3, 4, 5, 0])

    return tf.keras.Model(inputs=[inputs1], outputs=spatial_out_list)


def Generator_correct_spatial(sample_z, sample_t):
    inputs1 = tf.keras.layers.Input(shape=[256, 256, sample_z, 1, sample_t])
    initializer = tf.random_normal_initializer(0., 0.02)

    conv1_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu1_1 = tf.keras.layers.ReLU()

    conv1_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_2 = tf.keras.layers.BatchNormalization()
    relu1_2 = tf.keras.layers.ReLU()

    conv1_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_3 = tf.keras.layers.BatchNormalization()
    relu1_3 = tf.keras.layers.ReLU()

    conv1_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_4 = tf.keras.layers.BatchNormalization()
    relu1_4 = tf.keras.layers.ReLU()

    conv1_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_5 = tf.keras.layers.BatchNormalization()
    relu1_5 = tf.keras.layers.ReLU()

    conv1_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_6 = tf.keras.layers.BatchNormalization()
    relu1_6 = tf.keras.layers.ReLU()

    conv1_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_7 = tf.keras.layers.BatchNormalization()
    relu1_7 = tf.keras.layers.ReLU()

    conv1_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_8 = tf.keras.layers.BatchNormalization()
    relu1_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv_trans3_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_1 = tf.keras.layers.Dropout(0.5)
    batchnorm3_1 = tf.keras.layers.BatchNormalization()
    relu3_1 = tf.keras.layers.ReLU()
    cat3_1 = tf.keras.layers.Concatenate()

    conv_trans3_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_2 = tf.keras.layers.Dropout(0.5)
    batchnorm3_2 = tf.keras.layers.BatchNormalization()
    relu3_2 = tf.keras.layers.ReLU()
    cat3_2 = tf.keras.layers.Concatenate()

    conv_trans3_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_3 = tf.keras.layers.Dropout(0.5)
    batchnorm3_3 = tf.keras.layers.BatchNormalization()
    relu3_3 = tf.keras.layers.ReLU()
    cat3_3 = tf.keras.layers.Concatenate()

    conv_trans3_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_4 = tf.keras.layers.Dropout(0.5)
    batchnorm3_4 = tf.keras.layers.BatchNormalization()
    relu3_4 = tf.keras.layers.ReLU()
    cat3_4 = tf.keras.layers.Concatenate()

    conv_trans3_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_5 = tf.keras.layers.Dropout(0.5)
    batchnorm3_5 = tf.keras.layers.BatchNormalization()
    relu3_5 = tf.keras.layers.ReLU()
    cat3_5 = tf.keras.layers.Concatenate()

    conv_trans3_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_6 = tf.keras.layers.Dropout(0.5)
    batchnorm3_6 = tf.keras.layers.BatchNormalization()
    relu3_6 = tf.keras.layers.ReLU()
    cat3_6 = tf.keras.layers.Concatenate()

    conv_trans3_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_7 = tf.keras.layers.Dropout(0.5)
    batchnorm3_7 = tf.keras.layers.BatchNormalization()
    relu3_7 = tf.keras.layers.ReLU()
    cat3_7 = tf.keras.layers.Concatenate()

    conv_trans3_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

    # --------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    def encoder1(inputs):
        skips = []
        x = conv1_1(inputs)
        x = relu1_1(x)
        skips.append(x)

        x = conv1_2(x)
        x = batchnorm1_2(x)
        x = relu1_2(x)
        skips.append(x)

        x = conv1_3(x)
        x = batchnorm1_3(x)
        x = relu1_3(x)
        skips.append(x)

        x = conv1_4(x)
        x = batchnorm1_4(x)
        x = relu1_4(x)
        skips.append(x)

        x = conv1_5(x)
        x = batchnorm1_5(x)
        x = relu1_5(x)
        skips.append(x)

        x = conv1_6(x)
        x = batchnorm1_6(x)
        x = relu1_6(x)
        skips.append(x)

        x = conv1_7(x)
        x = batchnorm1_7(x)
        x = relu1_7(x)
        skips.append(x)

        x = conv1_8(x)
        x = batchnorm1_8(x)
        x = relu1_8(x)

        return x, skips


    def decoder1(inputs, skips):
        x = conv_trans3_1(inputs)
        x = batchnorm3_1(x)
        x = drop3_1(x)
        x = relu3_1(x)
        x = cat3_1([x, skips[6]])

        x = conv_trans3_2(x)
        x = batchnorm3_2(x)
        x = drop3_2(x)
        x = relu3_2(x)
        x = cat3_2([x, skips[5]])

        x = conv_trans3_3(x)
        x = batchnorm3_3(x)
        x = drop3_3(x)
        x = relu3_3(x)
        x = cat3_3([x, skips[4]])

        x = conv_trans3_4(x)
        x = batchnorm3_4(x)
        x = relu3_4(x)
        x = cat3_4([x, skips[3]])

        x = conv_trans3_5(x)
        x = batchnorm3_5(x)
        x = relu3_5(x)
        x = cat3_5([x, skips[2]])

        x = conv_trans3_6(x)
        x = batchnorm3_6(x)
        x = relu3_6(x)
        x = cat3_6([x, skips[1]])

        x = conv_trans3_7(x)
        x = batchnorm3_7(x)
        x = relu3_7(x)
        x = cat3_7([x, skips[0]])

        x = conv_trans3_last(x)
        return x


    # Downsampling through the model
    spatial_out_list = []
    skips1_list = []
    conved_output1_list = []

    # with tf.compat.v1.variable_scope('scope1', reuse=True):
    for t in range(0, inputs1.shape[5]):
        conved_output1, skips1_1 = encoder1(inputs1[:, :, :, :, :, t])
        spatial_out = decoder1(conved_output1, skips1_1)
        spatial_out_list.append(spatial_out)

    spatial_out_list = tf.convert_to_tensor(spatial_out_list)
    spatial_out_list = tf.transpose(spatial_out_list, [1, 2, 3, 4, 5, 0])


    return tf.keras.Model(inputs=[inputs1], outputs=spatial_out_list)

def Generator_correct_temporal():
    inputs1 = tf.keras.layers.Input(shape=[256, 256, 10, 1, 16])
    inputs2 = tf.keras.layers.Input(shape=[256, 256, 10, 1, 16])
    initializer = tf.random_normal_initializer(0., 0.02)

    conv1_1 = tf.keras.layers.Conv3D(32, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    relu1_1 = tf.keras.layers.ReLU()

    conv1_2 = tf.keras.layers.Conv3D(64, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_2 = tf.keras.layers.BatchNormalization()
    relu1_2 = tf.keras.layers.ReLU()

    conv1_3 = tf.keras.layers.Conv3D(128, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_3 = tf.keras.layers.BatchNormalization()
    relu1_3 = tf.keras.layers.ReLU()

    conv1_4 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_4 = tf.keras.layers.BatchNormalization()
    relu1_4 = tf.keras.layers.ReLU()

    conv1_5 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_5 = tf.keras.layers.BatchNormalization()
    relu1_5 = tf.keras.layers.ReLU()

    conv1_6 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_6 = tf.keras.layers.BatchNormalization()
    relu1_6 = tf.keras.layers.ReLU()

    conv1_7 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_7 = tf.keras.layers.BatchNormalization()
    relu1_7 = tf.keras.layers.ReLU()

    conv1_8 = tf.keras.layers.Conv3D(256, 4, strides=[2, 2, 1], padding='same',
                                     kernel_initializer=initializer, use_bias=False)
    batchnorm1_8 = tf.keras.layers.BatchNormalization()
    relu1_8 = tf.keras.layers.ReLU()

    # --------------------------------------------------------------------------------------------------------------
    conv_trans3_1 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_1 = tf.keras.layers.Dropout(0.5)
    batchnorm3_1 = tf.keras.layers.BatchNormalization()
    relu3_1 = tf.keras.layers.ReLU()
    cat3_1 = tf.keras.layers.Concatenate()

    conv_trans3_2 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_2 = tf.keras.layers.Dropout(0.5)
    batchnorm3_2 = tf.keras.layers.BatchNormalization()
    relu3_2 = tf.keras.layers.ReLU()
    cat3_2 = tf.keras.layers.Concatenate()

    conv_trans3_3 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_3 = tf.keras.layers.Dropout(0.5)
    batchnorm3_3 = tf.keras.layers.BatchNormalization()
    relu3_3 = tf.keras.layers.ReLU()
    cat3_3 = tf.keras.layers.Concatenate()

    conv_trans3_4 = tf.keras.layers.Conv3DTranspose(256, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_4 = tf.keras.layers.Dropout(0.5)
    batchnorm3_4 = tf.keras.layers.BatchNormalization()
    relu3_4 = tf.keras.layers.ReLU()
    cat3_4 = tf.keras.layers.Concatenate()

    conv_trans3_5 = tf.keras.layers.Conv3DTranspose(128, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_5 = tf.keras.layers.Dropout(0.5)
    batchnorm3_5 = tf.keras.layers.BatchNormalization()
    relu3_5 = tf.keras.layers.ReLU()
    cat3_5 = tf.keras.layers.Concatenate()

    conv_trans3_6 = tf.keras.layers.Conv3DTranspose(64, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_6 = tf.keras.layers.Dropout(0.5)
    batchnorm3_6 = tf.keras.layers.BatchNormalization()
    relu3_6 = tf.keras.layers.ReLU()
    cat3_6 = tf.keras.layers.Concatenate()

    conv_trans3_7 = tf.keras.layers.Conv3DTranspose(32, 4, strides=[2, 2, 1],
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    use_bias=False)
    drop3_7 = tf.keras.layers.Dropout(0.5)
    batchnorm3_7 = tf.keras.layers.BatchNormalization()
    relu3_7 = tf.keras.layers.ReLU()
    cat3_7 = tf.keras.layers.Concatenate()

    conv_trans3_last = tf.keras.layers.Conv3DTranspose(1, 4,
                                                       strides=[2, 2, 1],
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       activation='tanh')  # (batch_size, 256, 256, 3)

    # --------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    def encoder1(inputs):
        skips = []
        x = conv1_1(inputs)
        x = relu1_1(x)
        skips.append(x)

        x = conv1_2(x)
        x = batchnorm1_2(x)
        x = relu1_2(x)
        skips.append(x)

        x = conv1_3(x)
        x = batchnorm1_3(x)
        x = relu1_3(x)
        skips.append(x)

        x = conv1_4(x)
        x = batchnorm1_4(x)
        x = relu1_4(x)
        skips.append(x)

        x = conv1_5(x)
        x = batchnorm1_5(x)
        x = relu1_5(x)
        skips.append(x)

        x = conv1_6(x)
        x = batchnorm1_6(x)
        x = relu1_6(x)
        skips.append(x)

        x = conv1_7(x)
        x = batchnorm1_7(x)
        x = relu1_7(x)
        skips.append(x)

        x = conv1_8(x)
        x = batchnorm1_8(x)
        x = relu1_8(x)

        return x, skips


    def decoder1(inputs, skips):
        x = conv_trans3_1(inputs)
        x = batchnorm3_1(x)
        x = drop3_1(x)
        x = relu3_1(x)
        x = cat3_1([x, skips[6]])

        x = conv_trans3_2(x)
        x = batchnorm3_2(x)
        x = drop3_2(x)
        x = relu3_2(x)
        x = cat3_2([x, skips[5]])

        x = conv_trans3_3(x)
        x = batchnorm3_3(x)
        x = drop3_3(x)
        x = relu3_3(x)
        x = cat3_3([x, skips[4]])

        x = conv_trans3_4(x)
        x = batchnorm3_4(x)
        x = relu3_4(x)
        x = cat3_4([x, skips[3]])

        x = conv_trans3_5(x)
        x = batchnorm3_5(x)
        x = relu3_5(x)
        x = cat3_5([x, skips[2]])

        x = conv_trans3_6(x)
        x = batchnorm3_6(x)
        x = relu3_6(x)
        x = cat3_6([x, skips[1]])

        x = conv_trans3_7(x)
        x = batchnorm3_7(x)
        x = relu3_7(x)
        x = cat3_7([x, skips[0]])

        x = conv_trans3_last(x)
        return x


    # Downsampling through the model

    temporal_out_list = []

    # with tf.compat.v1.variable_scope('scope1', reuse=True):
    for t in range(0, inputs1.shape[5]):
        conved_output1, skips1_1 = encoder1(inputs1[:, :, :, :, :, t])
        temporal_out = decoder1(conved_output1, skips1_1)
        temporal_out_list.append(temporal_out)

    temporal_out_list = tf.convert_to_tensor(temporal_out_list)
    temporal_out_list = tf.transpose(temporal_out_list, [1, 2, 3, 0, 5, 4])


    return tf.keras.Model(inputs=[inputs1, inputs2], outputs=temporal_out_list)

def Discriminator_full():
  inp1 = tf.keras.layers.Input(shape=[256, 256, 16, 10], name='input_image1')
  inp2 = tf.keras.layers.Input(shape=[256, 256, 16, 10], name='input_image2')
  x = tf.keras.layers.concatenate([inp1, inp2])  # (bs, 256, 256, channels*2)
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

  last = tf.keras.layers.Conv3D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  # return tf.keras.Model(inputs=[inp1, inp2, tar1, tar2], outputs=last)
  return tf.keras.Model(inputs=[inp1, inp2], outputs=[last, down1, down2, down3, last])

def Discriminator_zfull(sample_z, sample_t):
  inp1 = tf.keras.layers.Input(shape=[256, 256, sample_z*sample_t, 1], name='input_image1')
  inp2 = tf.keras.layers.Input(shape=[256, 256, sample_z*sample_t, 1], name='input_image2')
  x = tf.keras.layers.concatenate([inp1, inp2])  # (bs, 256, 256, channels*2)
  initializer = tf.random_normal_initializer(0., 0.02)

  def downconv(inputs, filters, size, apply_batchnorm=True):
      x = tf.keras.layers.Conv3D(filters, size, strides=[2, 2, 2], padding='same',
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

  last = tf.keras.layers.Conv3D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  # return tf.keras.Model(inputs=[inp1, inp2, tar1, tar2], outputs=last)
  return tf.keras.Model(inputs=[inp1, inp2], outputs=[last, down1, down2, down3, last])