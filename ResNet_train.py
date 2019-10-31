# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:50:24 2019

@author: bjlij
"""

#import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from PIL import Image
import numpy as np
import keras.backend as K
#from keras.layers import Input, Lambda
#from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import ResNet

#import cv2
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_random_data(annotation_line):
    """random preprocessing for real-time data augmentation"""
    line = annotation_line.split()
    #image = Image.open((line[0]+' '+line[1])[1:-1])
    image = Image.open(line[0])
    #img = image.crop([120, 165, 1700, 955]).resize([1600,800])
    img_np = np.array(image)
    #img_np = np.where(img_np>=150,255,0).astype(np.uint8)
    #weight = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    #img_small = Image.fromarray(cv2.filter2D(img_np,-1,weight)).resize([400,400])
    #_img = np.array(img_small)
    '''
    for i in range(img_np.shape[1]-1, img_np.shape[1]-30, -1):
        for k in range(img_np.shape[0]):
            if np.average(img_np[k,i])<250:
                img_np[k,i]=0,0,0        
            if np.average(img_np[:,i])>50:
                img_np[:,i]=0
    '''
    return img_np, float(line[1])



def data_generator(annotation_lines, batch_size):
    """data generator for fit_generator"""
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        y_true = []
        for b in range(batch_size):
            if i == 0:
                pass
                #np.random.shuffle(annotation_lines)
            img_np, y = get_random_data(annotation_lines[i])
            image_data.append(img_np/255.)
            y_true.append(y)
            i = (i+1) % n
        image_data = np.array(image_data)
        y_true = to_categorical(y_true, 3)
        #yield image_data, y_true
        yield image_data


def data_generator_wrapper(annotation_lines, batch_size):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size)


def _main():
    annotation_path = '../2019.txt'
    log_dir = '../logs/000/'

    model = ResNet.ResNet.build(400, 400, 3, 3, [3, 3, 3], [64, 128, 256, 512])
    #model.load_weights(log_dir + 'ep032-loss0.613-val_loss0.609.h5')
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-acc{acc:.3f}-val_acc{val_acc:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    train_split = 1
    val_split = 0.3
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_valtrain = int(len(lines)*train_split)
    num_val = int(num_valtrain*val_split)
    num_train = num_valtrain - num_val
    num_test = len(lines) - num_valtrain
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        #for i in range(len(model.layers)):
            #model.layers[i].trainable = True
        #model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])  # recompile to apply the change
        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])  # recompile to apply the change
        #print('Unfreeze all of the layers.')

        batch_size = 100  # note that more GPU memory is required after unfreezing the body
        print('Totle samples {},Train on {} samples,val on {} samples, test on {} samples, with batch size {}.'.format(len(lines), num_train, num_valtrain - num_train, num_test, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:num_valtrain], batch_size),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=100,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint],
                            verbose=1)
        model.save_weights(log_dir + 'trained_weights_final.h5')
    #loss,accuracy = test(model, lines[num_valtrain:], batch_size)
    #print(loss,accuracy)
    return model,lines

def test(model,annotation_lines, batch_size):
    return model.evaluate_generator(data_generator_wrapper(annotation_lines, batch_size),steps=max(1, len(annotation_lines)//batch_size))


#model,lines = _main()
