from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from keras.models import load_model
import tensorflow as tf
import os,time,cv2, sys, math, glob, gc
import scipy
import argparse
import random

from module.util.load_data import loadDataGeneral
import module.model.RCycleGAN3D as RCycleGAN3D
import module.model.DualAxisNet as DualAxisNet


def saggital(img):
    """Extracts midle layer in saggital axis and rotates it appropriately."""
    return img[:,  int(img.shape[1] / 2), ::-1].T


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def decode_img(img):
  # img = tf.image.decode_jpeg(img, channels=1) #color images
  img = tf.io.decode_image(img, channels=1)
  img = tf.image.convert_image_dtype(img, tf.float32)
  size = tf.shape(img)
   #convert unit8 tensor to floats in the [0,1]range
  return img
#resize the image into 224*224

def process_path(file_path):
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img


def prepare_for_training(ds, batch, cache=True, shuffle_buffer_size=10):
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.repeat()  # repeat forever
    ds = ds.batch(batch)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def loadDataGeneral3D(path):
    """
    This function loads data stored in nifti format. Data should already be of
    appropriate shape.

    Inputs:
    - df: Pandas dataframe with two columns: image filenames and ground truth filenames.
    - path: Path to folder containing filenames from df.
    - append_coords: Whether to append coordinate channels or not.
    Returns:
    - X: Array of 3D images with 1 or 4 channels depending on `append_coords`.
    - y: Array of 3D masks with 1 channel.
    """
    X= []
    for file in path:
        img = nib.load(file).get_data()/255.0
        # size=img.shape
        X.append(img)

    X = np.float32(X)
    X = np.expand_dims(X, -1)

    print('### Dataset loaded')
    print('\t{}'.format(path))
    return X

def make_data(file, batch_size):
    # make the loaded np data as a tensorflow dataset
    train_dataset = loadDataGeneral3D(file)
    print(train_dataset[0].shape)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(args.batch_size, reshuffle_each_iteration=True)
    return train_dataset


def calculate_fid(act1, act2):
    # the evaluation metric FID
    # calculate activations
    # calculate mean and covariance statistics
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act1, axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

parser = argparse.ArgumentParser()
parser.add_argument('--class_balancing', type=str2bool, default=True, help='Whether to use median frequency class weights to balance the classes in the loss')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=2, help='How often to perform validation (epochs)')
parser.add_argument('--input_c_dim', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--data_dim', type=int, default=3, help='Sample dimension')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--dataset1', type=str, default="Myo_Ecad/10_256_contrast", help='Dataset you are using: Aju_Ecad, Myo_Ecad, Ecad_Myo, Myo_Aju, Aju_Myo')
parser.add_argument('--pdataset', type=str, default="Myo_Ecad_contrast/5_t1_256/06", help='Prediction dataset you are using.')
parser.add_argument('--clip_z', type=int, default=16, help='the length of z-axis in sample clip. 16 for Ecad-Myo, 6 for Aju-Myo')
parser.add_argument('--clip_t', type=int, default=10, help='the length of t-axis in sample clip. 10')
parser.add_argument('--start_epochs', type=int, default=0, help='the start training epoch')
parser.add_argument('--end_epochs', type=int, default=100, help='the stop training epoch')
parser.add_argument('--mode', type=str, default="train", help='Select "train", "test", or "predict" mode. \
Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--model', type=str, default="DualAxisNet_fold", help='Select "DualAxisNet_fold", "DualAxisNet_fold_guide2".')


args = parser.parse_args()
if __name__ == '__main__':

    network = None
    # for 3d volume, save z1-z6 in validation stage
    img_save_depth_a = 1
    img_save_depth_b = 6

    if args.mode == "train":

        # Load the data
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # load the list of sample names, and shuffle
        print("Loading the data from 3D dataset: %s" % (args.dataset1))
        train_filename1 = glob.glob('data/train_data/' + args.dataset1 + '_train/*.nii')
        val_filename1 = glob.glob('data/train_data/' + args.dataset1 + '_val/*.nii')
        print("length of training data: %s" % (str(len(train_filename1))))
        print("length of val data: %s" % (str(len(val_filename1))))
        random.shuffle(train_filename1)
        random.shuffle(val_filename1)

        # split the train and val samples to small groups (to avoid RAN overflow)
        train_filename_list=[]
        for n in range(0,len(train_filename1)//80):
            train_filename_list.append(train_filename1[n*80 :(n+1)*80])
        train_filename_list.append(train_filename1[len(train_filename1)// 80 * 80 : len(train_filename1)])

        val_filename_list = []
        for n in range(0, len(val_filename1) // 80):
            val_filename_list.append(val_filename1[n * 80:(n + 1) * 80])
        val_filename_list.append(val_filename1[len(val_filename1) // 80 * 80: len(val_filename1)])


        # Do the training here
        start_time = time.time()

        # build the G and D
        generator_t = DualAxisNet.Generator_correct_connect_fold(sample_z=args.clip_z, sample_t=args.clip_t)
        discriminator_Bt = DualAxisNet.Discriminator_zfull(sample_z=args.clip_z, sample_t=args.clip_t)

        generator_t.summary()

        LAMBDA = 100

        # define loss functions
        def generator_l1_loss(gen_output_2, target_2):
            # loss = RCycleGAN3D.l1_loss(gen_output_2, target_2) + RCycleGAN3D.cosinesimilarity_loss(gen_output_2, target_2)
            loss = tf.reduce_mean(tf.abs(target_2 - gen_output_2))
            # loss = tf.reduce_mean(tf.nn.l2_loss(target_2 - gen_output_2))
            return loss
        def generator_l2_loss(gen_output_2, target_2):
            # loss = RCycleGAN3D.l1_loss(gen_output_2, target_2) + RCycleGAN3D.cosinesimilarity_loss(gen_output_2, target_2)
            # loss = tf.reduce_mean(tf.abs(target_2 - gen_output_2))
            loss = tf.reduce_mean(tf.nn.l2_loss(target_2 - gen_output_2))
            return loss

        def generator_loss_adversarial(disc_A_real_output_1, disc_A_generated_output_1):
            loss = RCycleGAN3D.binarycrossentroty_loss_G(disc_A_real_output_1, disc_A_generated_output_1)# + RCycleGAN2D.binarycrossentroty_loss_G(disc_B_real_output_1, disc_B_generated_output_1)
            return loss


        def discriminator_loss(disc_real_output, disc_generated_output):
            loss1 = RCycleGAN3D.binarycrossentroty_loss_D(disc_real_output, disc_generated_output)
            return loss1
        def calc_cycle_loss(real_image, cycled_image):
            loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

            return loss1

        def ssim_loss(fake_image, real_image):
            # loss_rec = tf.reduce_mean(tf.image.ssim_multiscale(real_image, fake_image, max_val=1.0, filter_size=4))
            loss_rec = tf.reduce_mean(tf.image.ssim(real_image, fake_image, 1.0))
            return -1 * loss_rec

        def ssimdiff_loss(fake1_image, real1_image, real2_image):
            loss_rec = tf.reduce_mean(tf.image.ssim(real2_image, fake1_image, max_val=1.0)) - tf.reduce_mean(tf.image.ssim(real1_image, fake1_image, max_val=1.0))
            return loss_rec
        def perceptual_loss(fake_image, real_image):
            loss = tf.reduce_mean(tf.abs(fake_image - real_image))
            return loss

        def frame_loss(fake_image, real_image):
            loss = []
            # print(fake_image.shape, real_image.shape)
            for t in range(0, fake_image.shape[5]-1):
                loss.append(tf.reduce_mean(tf.abs((real_image[:,:,:,:,:,t+1]-real_image[:,:,:,:,:,t]) - (fake_image[:,:,:,:,:,t+1]-fake_image[:,:,:,:,:,t]))))
            return tf.reduce_mean(loss)

        def ssim_loss_t(fake_image, real_image):
            loss = []
            # loss_rec = tf.reduce_mean(tf.image.ssim_multiscale(real_image, fake_image, max_val=1.0, filter_size=4))
            for t in range(0, fake_image.shape[5] - 1):
                loss.append(tf.reduce_mean(tf.image.ssim(real_image[:,:,:,:,:,t], fake_image[:,:,:,:,:,t], 1.0)))
            return -1 * tf.reduce_mean(loss)


        # initialize the training optimizer
        initial_learning_rate = 2e-4 * (0.96**((len(train_filename1)*args.start_epochs)//1000))
        # initial_learning_rate = 2e-4
        print("initial_learning_rate=",initial_learning_rate)
        lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-4,
            decay_steps=200,
            decay_rate=0.96,
            staircase=True)

        lr_schedule2 = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,#len(train_filename1),
            decay_rate=0.96,
            staircase=True)

        generator_t_optimizer = tf.keras.optimizers.Adam(lr_schedule2, beta_1=0.5)

        discriminator_Bt_optimizer = tf.keras.optimizers.Adam(lr_schedule2, beta_1=0.5)


        # build the checkpoint
        checkpoint_dir4 = './training_checkpoints' + '/' + args.model + '/' + args.dataset1
        checkpoint_prefix4 = os.path.join(checkpoint_dir4, "ckpt")
        checkpoint4 = tf.train.Checkpoint(
            generator_t=generator_t,
            # discriminator_e=discriminator_e,
            # discriminator_e_forloss=discriminator_e_forloss,
            discriminator_Bt=discriminator_Bt,
            generator_t_optimizer=generator_t_optimizer,
            # discriminator_e_optimizer=discriminator_e_optimizer,
            # discriminator_e_forloss_optimizer=discriminator_e_forloss_optimizer,
            discriminator_Bt_optimizer=discriminator_Bt_optimizer,
        )
        manager4 = tf.train.CheckpointManager(
            checkpoint4, directory=checkpoint_dir4, max_to_keep=1)

        if args.continue_training is True:
            status = checkpoint4.restore(manager4.latest_checkpoint).expect_partial()
            if status.assert_existing_objects_matched():
                print("Load checkpoint successfully")


        # inceptionv3 mode for FID calculation in val
        base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(256, 256, 3),
                                                                    include_top=False,
                                                                    pooling='avg')
        import datetime

        log_dir = "logs/"

        summary_writer = tf.summary.create_file_writer(
            log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


        # build the train step (loss calculation, gradient descent, G and D update)
        @tf.function
        def train_step_t(input_z_list, target_z_list, epoch):
            with tf.GradientTape() as gen_z_tape, tf.GradientTape() as gen_t_tape, tf.GradientTape() as disc_e_tape2, tf.GradientTape() as disc_e_tape, tf.GradientTape() as disc_Bt_tape:
                # transform the list of video frames to a 6D tensor (with a dimension of batch)
                # transpose the tensor to an demanded view
                input_z_list = tf.convert_to_tensor(input_z_list)
                input_z_list = tf.transpose(input_z_list, [1,2,3,4,5,0])

                target_z_list = tf.convert_to_tensor(target_z_list)
                target_z_list = tf.transpose(target_z_list, [1, 2, 3, 4, 5, 0])

                # predict the output
                pred_a= generator_t(input_z_list, training=True)

                # calculate the l1 loss and ssim loss (not used)
                l1loss = 100 * generator_l1_loss(pred_a,target_z_list)
                # ssimloss = 10 * ssim_loss_t(pred_a, target_z_list)

                # reshape the tensor for D
                pred_a = tf.reshape(pred_a, [pred_a.shape[0], pred_a.shape[1], pred_a.shape[2],
                                             pred_a.shape[3] * pred_a.shape[5], pred_a.shape[4]])
                target_z_list = tf.reshape(target_z_list, [target_z_list.shape[0], target_z_list.shape[1], target_z_list.shape[2],
                                                           target_z_list.shape[3] * target_z_list.shape[5], target_z_list.shape[4]])
                input_z_list = tf.reshape(input_z_list, [input_z_list.shape[0], input_z_list.shape[1], input_z_list.shape[2],
                                                         input_z_list.shape[3] * input_z_list.shape[5], input_z_list.shape[4]])


                # get D intermediate feature maps and final output
                [disc_Bt_real_output1, featureA_t_real_1, featureA_t_real_2, featureA_t_real_3, _] = discriminator_Bt(
                    [input_z_list, target_z_list], training=True)
                [disc_Bt_fake_output1, featureA_t_fake_1, featureA_t_fake_2, featureA_t_fake_3, _] = discriminator_Bt(
                    [input_z_list, pred_a], training=True)

                # calculate perceptual loss using D feature maps
                perceptualloss = 100 * perceptual_loss(featureA_t_fake_1, featureA_t_real_1) #+ 100 * perceptual_loss(featureA_t_fake_2, featureA_t_real_2) + 100 * perceptual_loss(featureA_t_fake_3, featureA_t_real_3)
                # calculate adversarial loss using D final output
                generator_t_loss = generator_loss_adversarial(disc_Bt_real_output1, disc_Bt_fake_output1)
                # get the final objective for G
                generator_t_total_loss = generator_t_loss \
                                         + perceptualloss \
                                         + l1loss\

                # get the final objective for D
                disc_Bt_loss = discriminator_loss(disc_Bt_real_output1, disc_Bt_fake_output1)

                # gradient descent
                generator_t_gradients = gen_t_tape.gradient(generator_t_total_loss,
                                                             generator_t.trainable_variables)
                discriminator_Bt_gradients = disc_Bt_tape.gradient(disc_Bt_loss,
                                                                   discriminator_Bt.trainable_variables)

                # update networks
                generator_t_optimizer.apply_gradients(zip(generator_t_gradients,
                                                          generator_t.trainable_variables))
                discriminator_Bt_optimizer.apply_gradients(zip(discriminator_Bt_gradients,
                                                               discriminator_Bt.trainable_variables))

                # delate the variables to recycle RAM
                del input_z_list, target_z_list, pred_a, disc_Bt_real_output1, featureA_t_real_1, featureA_t_real_2, featureA_t_real_3
                del disc_Bt_fake_output1, featureA_t_fake_1, featureA_t_fake_2, featureA_t_fake_3
                gc.collect()

            with summary_writer.as_default():
                tf.summary.scalar('gen_t_loss', generator_t_loss, step=epoch)
                tf.summary.scalar('perceptual_loss', perceptualloss, step=epoch)
                tf.summary.scalar('l1_loss', l1loss, step=epoch)
                tf.summary.scalar('disc_Be_loss', disc_Bt_loss, step=epoch)
            return generator_t_loss, perceptualloss, l1loss, disc_Bt_loss


        # build a training loop
        def fit(start_epochs, end_epochs):
            for epoch in range(start_epochs, end_epochs+5):
                if not os.path.isdir("%s/%s/%s/%04d" % ("./training_checkpoints", args.model, args.dataset1, epoch)):
                    os.makedirs("%s/%s/%s/%04d" % ("./training_checkpoints", args.model, args.dataset1, epoch))
                start = time.time()
                frame_loss_list = []
                l1_loss_list = []
                gen_t_loss_list = []
                disc_Bz_loss_list = []
                disc_Bt_loss_list = []

                frame_loss_total_loss_list = []
                l1_loss_total_loss_list = []
                gen_t_total_loss_list = []
                disc_Bz_total_loss_list = []
                disc_Bt_total_loss_list = []

                val_MSE_me=[]
                val_SSIM_me = []
                val_PSNR_me = []
                val_MSE_video = []

                # Train
                print('Epoch: %04d, training of Myo and Ecad' % (epoch))
                # for each small group of sample, load and make a tensorflow dataset
                for train_filename1 in train_filename_list:
                    train_i1 = make_data(train_filename1, args.batch_size)
                    for n, image_batch in train_i1.enumerate():
                        if (n + 1) % 20 == 0:
                            print('%f - %f: gen_z loss = %f, perceptual_loss = %f, l1_loss = %f, Disc_Be loss = %f' %
                                  (n+1-20, n+1, np.mean(gen_t_loss_list), np.mean(frame_loss_list), np.mean(l1_loss_list), np.mean(disc_Bt_loss_list)))
                            frame_loss_list=[]
                            gen_t_loss_list=[]
                            l1_loss_list = []
                            disc_Bt_loss_list=[]

                        print(n)
                        input_z_list = []
                        target_z_list = []
                        # build a frame list from the input
                        for frame_n in range(0, image_batch.shape[2] // 256):
                            input_image1 = image_batch[:,
                                           0 * int(image_batch.shape[1] // 2):1 * int(image_batch.shape[1] // 2),
                                           frame_n * int(image_batch.shape[2] // 10):(frame_n + 1) * int(
                                               image_batch.shape[2] // 10), :, :]

                            target1 = image_batch[:,
                                      1 * int(image_batch.shape[1] // 2):2 * int(image_batch.shape[1] // 2),
                                      (frame_n) * int(image_batch.shape[2] // 10):(frame_n + 1) * int(
                                          image_batch.shape[2] // 10), :, :]

                            input_z_list.append(input_image1)
                            target_z_list.append(target1)

                        # training step
                        gen_t_total_loss, frame_loss, l1_loss, disc_Bt_loss = train_step_t(input_z_list, target_z_list, epoch)

                        # append losses for documentation
                        gen_t_loss_list.append(gen_t_total_loss)
                        l1_loss_list.append(l1_loss)
                        frame_loss_list.append(frame_loss)
                        disc_Bt_loss_list.append(disc_Bt_loss)

                        gen_t_total_loss_list.append(gen_t_total_loss)
                        l1_loss_total_loss_list.append(l1_loss)
                        frame_loss_total_loss_list.append(frame_loss)
                        disc_Bt_total_loss_list.append(disc_Bt_loss)

                        # relief RAM
                        del input_z_list, target_z_list, image_batch
                        gc.collect()
                    del train_i1
                    gc.collect()



                print('.', end='')
                print(
                    'Avg gen_total_loss_list = %f, Avg frame_loss = %f, Avg l1_loss = %f, Avg disc_Bt_total_loss_list = %f' %
                    (np.mean(gen_t_total_loss_list), np.mean(frame_loss_total_loss_list), np.mean(l1_loss_total_loss_list),
                     np.mean(disc_Bt_total_loss_list)))

                # save checkpoint
                if (epoch + 1) % args.checkpoint_step == 0:
                    print('saving checkpoint')
                    manager4.save()

                # val every two training epoch
                if epoch % args.validation_step == 0:
                    print('epoch: %f, validation' % (epoch))
                    Valfile = open(
                        "%s/%s/%s/%04d/val_scores.csv" % ("./training_checkpoints", args.model, args.dataset1, epoch),
                        'w')
                    Valfile.write(
                        " , val_MSE_me, val_SSIM_me, val_PSNR_me, val_MSE_video\n")

                    n=0
                    # for each small group of sample, load and make a tensorflow dataset
                    for val_filename1 in val_filename_list:
                        if n == 20: # only val 20 sample (FID takes too long)
                            break
                        val_i1 = make_data(val_filename1, 1)
                        for n, image_batch in val_i1.enumerate():
                            print(n)
                            input_z_list = []
                            target_z_list = []
                            for frame_n in range(0, image_batch.shape[2] // 256):
                                input_image1 = image_batch[:,
                                               0 * int(image_batch.shape[1] // 2):1 * int(image_batch.shape[1] // 2),
                                               frame_n * int(image_batch.shape[2] // 10):(frame_n + 1) * int(
                                                   image_batch.shape[2] // 10), :, :]

                                target1 = image_batch[:,
                                          1 * int(image_batch.shape[1] // 2):2 * int(image_batch.shape[1] // 2),
                                          (frame_n) * int(image_batch.shape[2] // 10):(frame_n + 1) * int(
                                              image_batch.shape[2] // 10), :, :]

                                input_z_list.append(input_image1)
                                target_z_list.append(target1)

                            input_z_list = tf.convert_to_tensor(input_z_list)
                            input_z_list = tf.transpose(input_z_list, [1, 2, 3, 4, 5, 0])
                            target_z_list = tf.convert_to_tensor(target_z_list)
                            target_z_list = tf.transpose(target_z_list, [1, 2, 3, 4, 5, 0])
                            # predict
                            pred_z = generator_t(input_z_list, training=False)

                            # run FID, the FID with Inception3 requires input size (2048, 2048)
                            reset_1 = 0
                            x_feature_list = []
                            y_feature_list = []
                            for t in range(pred_z.shape[5]):
                                if reset_1 == 128:
                                    break
                                for z in range(pred_z.shape[3]):
                                    a = target_z_list[:, :, :, z, :, t]
                                    b = pred_z[:, :, :, z, :, t]
                                    a = tf.tile(a, [1, 1, 1, 3])
                                    b = tf.tile(b, [1, 1, 1, 3])

                                    x_feature = base_model(a, training=False)
                                    y_feature = base_model(b, training=False)
                                    x_feature_list.append(x_feature)
                                    y_feature_list.append(y_feature)
                                    reset_1 += 1

                                    if reset_1 == 128:
                                        x_feature_list = tf.squeeze(tf.convert_to_tensor(x_feature_list))
                                        x_feature_list = tf.tile(x_feature_list, [16, 1])

                                        y_feature_list = tf.squeeze(tf.convert_to_tensor(y_feature_list))
                                        y_feature_list = tf.tile(y_feature_list, [16, 1])
                                        break

                            if reset_1<128:
                                x_feature_list = tf.squeeze(tf.convert_to_tensor(x_feature_list))
                                x_feature_list = tf.tile(x_feature_list, [2048//x_feature_list.shape[0], 1])
                                paddings = tf.constant([[(2048-x_feature_list.shape[0])//2, (2048-x_feature_list.shape[0])//2], [0, 0]])
                                x_feature_list = tf.pad(x_feature_list, paddings, mode='CONSTANT', constant_values=0)

                                y_feature_list = tf.squeeze(tf.convert_to_tensor(y_feature_list))
                                y_feature_list = tf.tile(y_feature_list, [2048 // y_feature_list.shape[0], 1])
                                paddings = tf.constant(
                                    [[(2048 - y_feature_list.shape[0]) // 2, (2048 - y_feature_list.shape[0]) // 2],
                                     [0, 0]])
                                y_feature_list = tf.pad(y_feature_list, paddings, mode='CONSTANT', constant_values=0)

                            val_MSE_me.append(calculate_fid(x_feature_list, y_feature_list))

                            # write predictions as images
                            for t in range(0, pred_z.shape[5]):
                                if int(n) % 20 == 0:
                                    fig = plt.figure(figsize=(25, 25))
                                    for depth in range(img_save_depth_a, img_save_depth_b):
                                        img = np.squeeze(np.array(input_z_list[:, :, :, depth, :, t]), axis=None)
                                        fig.add_subplot(3, 5, 1 + depth - img_save_depth_a)
                                        plt.imshow(img)

                                    for depth in range(img_save_depth_a, img_save_depth_b):
                                        img = np.squeeze(np.array(target_z_list[:, :, :, depth, :, t]), axis=None)
                                        fig.add_subplot(3, 5, 6 + depth - img_save_depth_a)
                                        plt.imshow(img)

                                    for depth in range(img_save_depth_a, img_save_depth_b):
                                        img = np.squeeze(np.array(pred_z[:, :, :, depth, :, t]), axis=None)
                                        fig.add_subplot(3, 5, 11 + depth - img_save_depth_a)
                                        plt.imshow(img)


                                    # for depth in range(3, 8):
                                    #     img = np.squeeze(np.array(pred_t[:, :, :, depth, :, t]), axis=None)
                                    #     fig.add_subplot(4, 5, 16 + depth - 3)
                                    #     plt.imshow(img)

                                    fig.savefig("%s/%s/%s/%04d/%s_me_val %s.png" % (
                                        "./training_checkpoints", args.model, args.dataset1, epoch, str(int(n)),
                                        str(t)))
                                    plt.close(fig)

                            if n == 20:
                                break

                            del input_z_list, target_z_list, pred_z, x_feature_list, y_feature_list, image_batch
                            gc.collect()
                        del val_i1
                        gc.collect()

                    print(
                        'val_MSE_me = %f, val_SSIM_me = %f, val_PSNR_me = %f, val_MSE_video = %f' % (
                        tf.reduce_mean(val_MSE_me), tf.reduce_mean(val_SSIM_me), tf.reduce_mean(val_PSNR_me),
                        tf.reduce_mean(val_MSE_video)))
                    Valfile.write("%s, %f, %f, %f, %f" % (
                    'val', np.mean(val_MSE_me), np.mean(val_SSIM_me), np.mean(val_PSNR_me), np.mean(val_MSE_video)))
                    Valfile.close()

                print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                   time.time() - start))
                print('Time rest is {} hr\n'.format((end_epochs - epoch - 1) *
                                                    (time.time() - start) / 3600))
                # checkpoint.save(file_prefix=checkpoint_prefix)
            # manager4.save()

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        fit(args.start_epochs, args.end_epochs)

    elif args.mode == "predict":
        print("Prediction: Loading the data from dataset: %s" % (args.pdataset))
        # filename1 = glob.glob('Predict_sample/'+ args.model + '/' + args.pdataset + '/*.nii')

        # Load test data
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        if args.data_dim==3:
            img_size = [256, 256, 16]
            print("Loading the data from dataset: %s" % (args.pdataset))
            filename1 = glob.glob('data/pred_data/'+ args.pdataset + '/*.nii')
            X_train, y_train = loadDataGeneral(filename1, img_size)
            print(X_train.shape)
        else:
            ValueError('data_dim should not be 2 or 3')
        print('Data loading finish. ', X_train.shape)
        X_train = tf.cast(X_train, tf.float32)
        y_train = tf.cast(y_train, tf.float32)
        predict_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        predict_dataset = predict_dataset.batch(1)

        def perceptual_loss(real_image, fake_image):
            loss = tf.reduce_sum((fake_image - real_image +1))
            return loss

        start_time = time.time()

        generator_t = DualAxisNet.Generator_correct_connect_fold(sample_z=args.clip_z, sample_t=args.clip_t)

        checkpoint_dir4 = './training_checkpoints' + '/' + args.model + '/' + args.dataset1
        checkpoint_prefix4 = os.path.join(checkpoint_dir4, "ckpt")
        checkpoint4 = tf.train.Checkpoint(
            generator_t=generator_t,
        )
        manager4 = tf.train.CheckpointManager(
            checkpoint4, directory=checkpoint_dir4, max_to_keep=1)

        status = checkpoint4.restore(manager4.latest_checkpoint).expect_partial()
        if status.assert_existing_objects_matched():
            print("Load checkpoint successfully")


        val_2D_FID = []

        val_frameloss = []
        if not os.path.isdir("%s/%s/%s/%s" % ("./Prediction_result", args.model, args.dataset1, args.pdataset)):
            os.makedirs("%s/%s/%s/%s" % ("./Prediction_result", args.model, args.dataset1, args.pdataset))

        base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(256, 256, 3),
                                                                    include_top=False,
                                                                    pooling='avg')

        input_list = []
        target_list = []
        for n, (input_image, target) in predict_dataset.enumerate():
            # print(input_image.shape, target.shape)
            print(int(n), '/', X_train.shape[0])
            if int(n) < 10:
                nn = '000' + str(int(n))
            elif int(n) < 100:
                nn = '00' + str(int(n))
            elif int(n) < 1000:
                nn = '0' + str(int(n))
            else:
                nn = str(int(n))

            input_list.append(input_image)
            target_list.append(target)
            if (n + 1) % 10 == 0:
                input_list = tf.convert_to_tensor(input_list)
                print(input_list.shape)
                input_list = tf.transpose(input_list, [1, 2, 3, 4, 5, 0])

                target_list = tf.convert_to_tensor(target_list)
                target_list = tf.transpose(target_list, [1, 2, 3, 4, 5, 0])

                pred_z= generator_t([input_list], training=False)

                pred_z = tf.cast(pred_z * 255, tf.uint8)
                target_list = tf.cast(target_list * 255, tf.uint8)
                input_list = tf.cast(input_list * 255, tf.uint8)

                reset_1 = 0
                x_feature_list = []
                y_feature_list = []
                for t in range(pred_z.shape[5]):
                    if reset_1 == 128:
                        break
                    for z in range(pred_z.shape[3]):
                        a = target_list[:, :, :, z, :, t]
                        b = pred_z[:, :, :, z, :, t]
                        a = tf.tile(a, [1, 1, 1, 3])
                        b = tf.tile(b, [1, 1, 1, 3])

                        x_feature = base_model(a, training=False)
                        y_feature = base_model(b, training=False)
                        x_feature_list.append(x_feature)
                        y_feature_list.append(y_feature)
                        reset_1 += 1

                        if reset_1 == 128:
                            x_feature_list = tf.squeeze(tf.convert_to_tensor(x_feature_list))
                            x_feature_list = tf.tile(x_feature_list, [16, 1])

                            y_feature_list = tf.squeeze(tf.convert_to_tensor(y_feature_list))
                            y_feature_list = tf.tile(y_feature_list, [16, 1])
                            break
                if reset_1 < 128:
                    x_feature_list = tf.squeeze(tf.convert_to_tensor(x_feature_list))
                    x_feature_list = tf.tile(x_feature_list, [2048 // x_feature_list.shape[0], 1])
                    paddings = tf.constant(
                        [[(2048 - x_feature_list.shape[0]) // 2, (2048 - x_feature_list.shape[0]) // 2], [0, 0]])
                    x_feature_list = tf.pad(x_feature_list, paddings, mode='CONSTANT', constant_values=0)

                    y_feature_list = tf.squeeze(tf.convert_to_tensor(y_feature_list))
                    y_feature_list = tf.tile(y_feature_list, [2048 // y_feature_list.shape[0], 1])
                    paddings = tf.constant(
                        [[(2048 - y_feature_list.shape[0]) // 2, (2048 - y_feature_list.shape[0]) // 2],
                         [0, 0]])
                    y_feature_list = tf.pad(y_feature_list, paddings, mode='CONSTANT', constant_values=0)

                val_2D_FID.append(calculate_fid(x_feature_list, y_feature_list))

                new_predz = nib.Nifti1Image(np.squeeze(np.array(pred_z)),
                                            affine=np.eye(4))
                # new_predt = nib.Nifti1Image(np.squeeze(np.array(pred_t)),
                #                             affine=np.eye(4))

                new_input_gty = nib.Nifti1Image(np.squeeze(np.array(target_list), axis=None),
                                                affine=np.eye(4))
                nib.save(new_predz,
                         "%s/%s/%s/%s/%s_%s_%s_predz.nii" % (
                             "./Prediction_result", args.model, args.dataset1, args.pdataset, args.pdataset[-2:], nn,
                             args.model))
                # nib.save(new_predt,
                #          "%s/%s/%s/%s/%s_%s_%s_predt.nii" % (
                #              "./Prediction_result", args.model, args.dataset1, args.pdataset, args.pdataset[-2:], nn,
                #              args.model))

                nib.save(new_input_gty,
                         "%s/%s/%s/%s/%s_%s_%s_gty.nii" % (
                             "./Prediction_result", args.model, args.dataset1, args.pdataset, args.pdataset[-2:], nn,
                             args.model))

                new_input_gtx = nib.Nifti1Image(np.squeeze(np.array(input_list), axis=None),
                                                affine=np.eye(4))
                nib.save(new_input_gtx,
                         "%s/%s/%s/%s/%s_%s_%s_gtx.nii" % (
                             "./Prediction_result", args.model, args.dataset1, args.pdataset, args.pdataset[-2:], nn,
                             args.model))

                input_list = []
                target_list = []

        Valfile = open(
            "%s/%s/%s/%s/val_scores.csv" % ("./Prediction_result", args.model, args.dataset1, args.pdataset), 'w')
        Valfile.write(
            " , 2D_FID\n")
        Valfile.write(
            "%s, %f % (
                'val', np.mean(val_2D_FID)))
        Valfile.close()
        print(
            '2D_FID: %f' % (
                np.mean(val_2D_FID)))
