# -*- coding: utf-8 -*-
"""
Created on 6 NOVEMBER 2021 

@author: ADRIAN CHAVARRO
UNIMILITAR

"""

"""
Path folder
"""
Path_File = r"/home/tecnoparqueneiva/RES_ADR/SYMTOM_PROYECT_01-02-2022/NOTEBOOKS"
MODELS_PATH = r"/home/tecnoparqueneiva/RES_ADR/SYMTOM_PROYECT_01-02-2022/MODELS"
OUTPUTH_PATH = r"/home/tecnoparqueneiva/RES_ADR/SYMTOM_PROYECT_01-02-2022/DATAOUT"
DATA_ARRAY_FOLDER=r"/home/tecnoparqueneiva/RES_ADR/SYMTOM_PROYECT_01-02-2022/DATASET"


"""Libraries
"""

import tensorflow as tf
import os
import sys
import cv2
import matplotlib.pyplot as plt

# from plot_model import plot_model
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import Lambda, Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD as optSGD
from tensorflow.keras.optimizers import  Adam  
from tensorflow.keras.optimizers import  Nadam 
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.efficientnet import EfficientNetB5
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
print(tf.__version__)

from tensorflow import keras
import time
import efficientnet.tfkeras


"""
LOAD NUMPY ARRAY ADRCOFFEMIX AUGENTED
"""

## Loading numpy training data
X_train = np.load(DATA_ARRAY_FOLDER+'/X_train.npy', allow_pickle=True)
y_train = np.load(DATA_ARRAY_FOLDER+'/y_train.npy', allow_pickle=True)
## Load numpy Validation data
X_valid = np.load(DATA_ARRAY_FOLDER+'/X_valid.npy', allow_pickle=True)
y_valid = np.load(DATA_ARRAY_FOLDER+'/y_valid.npy', allow_pickle=True)
## Load numpy Test data
X_test = np.load(DATA_ARRAY_FOLDER+'/X_test.npy', allow_pickle=True)
y_test = np.load(DATA_ARRAY_FOLDER+'/y_test.npy', allow_pickle=True)




"""
SET CLASSES
"""
MAP_RUST = {0:'CERCOSPORA', 1:'HEALTH', 2:'MINER',3:'PHOMA',4:'RUST'}


"""DEFINITION OF PLOT FUNCTION
"""
def plot_result(history = 'history', name_accuracy='sparse_categorical_accuracy', name_loss = 'loss'):
    #-----------------------------------------------------------
    # Recupera los resultados sobre los datos de entrenamiento y test
    # para cada época de entrenamiento
    #-----------------------------------------------------------
    acc      = history.history[ name_accuracy ]
    val_acc  = history.history[ 'val_'+name_accuracy ]
    loss     = history.history[ name_loss ]
    val_loss = history.history[ 'val_'+name_loss ]

    epochs   = range(len(acc)) # Obtiene el número de epocas

    #------------------------------------------------
    # Gráfica de precisión de entrenamiento y validación por  epoch
    #------------------------------------------------
    plt.rcParams["figure.figsize"] = (8, 4)
    fig, axes = plt.subplots(ncols=2, nrows=1)
    ax1, ax2 = axes.ravel()

    # plt.subplot(211)
    ax1.plot  ( epochs,     acc, label='Entrenamiento')
    ax1.plot  ( epochs, val_acc, label='Validación')
    ax1.set_title ('Precisión Entrenamiento y validación')
    ax1.legend()
    #ax1.figure()

    #------------------------------------------------
    # Gráfica de pérdida (loss) entrenamiento y validación por época
    #------------------------------------------------
    # plt.subplot(212)
    ax2.plot  ( epochs,     loss, label='Entrenamiento')
    ax2.plot  ( epochs, val_loss, label='Validación')
    ax2.legend()
    ax2.set_title ('Pérdida en entrenamiento y validación')
    
    
    
    
#initial parameters for VGG19
#Lr=0.001
#Fc_size=256
#FZ_point='block5_pool'
#Epoch_length=50
    
#
##initial parameters for VGG19
#def initial_parameters():
Lr_m=[0.00001]
Fc_size_m=[1024]
FZ_point_m=['block5_pool'] #['block3_pool','block4_pool','block5_pool']
Epoch_length=50
Opt_m =['Nadam','Rmsprop']
# keras_models=['VGG19','Xception','ResNet50','Inceptionv3',
#               'MobileNetv2','Densenet201','EfficientNetb5','InceptionResnetv2']
# models_layers_name=['block5_pool','block14_sepconv2_act','conv5_block3_add',
#                     'mixed9','block_15_add','conv5_block32_concat',
#                     'top_conv,'block35_1_mixed']
keras_models=['Inceptionv3','Densenet201']

# models_layers_name=['block_8_10_mixed']
models_layers_name=['mixed9', 'conv5_block32_concat']

#    
#CREATE EXCEL FILE TO WRITE RESULTS
# import openpyxl
# from openpyxl import workbook
# wb1=workbook()
# ws1=wb1.active()

# wb2=workbook()
# ws2=wb2.active()


count_mod=0 
#def execute_training_models():
for modelo,capa in zip(keras_models,models_layers_name): 
    for Lr in Lr_m:
        print (Lr)
        for Fc_size in Fc_size_m:
            print(Fc_size)
            for FZ_point in FZ_point_m:
                print(FZ_point)
                for Opt_sel in Opt_m:
                    print(Opt_sel)
                    """
                    MODEL GENERATE
                    """
                    count_mod= count_mod+1
                    IMG_SIZE=224
                    x_in=IMG_SIZE
                    y_in=IMG_SIZE
                    
                    print("*********************************TEST MODEL:***********************************************************")
                    print("\m"+"_"+"symptom" + str(count_mod)+"-"+str(Lr)+"-"+str(Fc_size)+"-"+str(FZ_point) + "-"+ Opt_sel)
                    print(" /n")
                    if modelo=='VGG19':    
                        from tensorflow.keras.applications.vgg19 import VGG19
                        pre_trained_model= tf.keras.applications.VGG19(input_shape = (224, 224, 3), 
                                                                        include_top = False, 
                                                                        weights = 'imagenet')   
                        
                    if modelo=='Xception':    
                        from tensorflow.keras.applications.xception import Xception
                        pre_trained_model = tf.keras.applications.Xception(input_shape = (224, 224, 3), 
                                                                        include_top = False, 
                                                                        weights = 'imagenet')    
                    if modelo=='ResNet50': 
                        from tensorflow.keras.applications.resnet import ResNet50
                        pre_trained_model= tf.keras.applications.ResNet50(input_shape = (224, 224, 3), 
                                                                        include_top = False, 
                                                                        weights = 'imagenet')              
                    if modelo=='InceptionResnetv2':
                        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
                        pre_trained_model = tf.keras.applications.InceptionResNetV2(input_shape = (224, 224, 3), 
                                                                        include_top = False, 
                                                                        weights = 'imagenet')   
                    if modelo=='EfficientNetb5':
                        from tensorflow.keras.applications.efficientnet import EfficientNetB5
                        pre_trained_model= tf.keras.applications.efficientnet.EfficientNetB5(input_shape = (224, 224, 3),include_top = False,weights = 'imagenet') 
                        
                    if modelo=='Densenet201': 
                        from tensorflow.keras.applications.densenet import DenseNet201
                        pre_trained_model = tf.keras.applications.DenseNet201(input_shape = (224, 224, 3), 
                                                                        include_top = False, 
                                                                        weights = 'imagenet')    
                        
                    if modelo=='MobileNetv2':
                        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
                        pre_trained_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3), 
                                                                        include_top = False, 
                                                                        weights = 'imagenet')              
                    
                    if modelo=='Inceptionv3':
                        from tensorflow.keras.applications.inception_v3 import InceptionV3
                        pre_trained_model = tf.keras.applications.InceptionV3(input_shape = (224, 224, 3), 
                                                                        include_top = False, 
                                                                        weights = 'imagenet')              
              
    
    #                from tensorflow.keras.layers import Dropout
    #                from tensorflow.keras.regularizers import l2
                    
    #                from keras.applications.vgg19 import VGG19
    #                
    #                # Se apaga la red fully-connected y los pesos
    #                pre_trained_model = tf.keras.applications.VGG19(input_shape = (x_in, y_in, 3), 
    #                                                include_top = False, 
    #                                                weights = 'imagenet')
    #                # Se presenta el resumen del modelo.  Se bloquean las capas para no entrenarse
                    for layer in pre_trained_model.layers:
                        layer.trainable = False
                    
    #                pre_trained_model.summary()
                    
                    #plot_model(pre_trained_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
                    # Se selecciona una capa del modelo , hasta la cual se toma la transferencia
                    last_layer = pre_trained_model.get_layer(capa) 
                    print('last layer output shape: ', last_layer.output_shape)
                    # Se asigna la última capa seleccionada
                    last_output = last_layer.output
                    last_output
                    
                    # Se adiciona el fully-connected nuevo (de nuestro modelo)
                    
                    ##### Neural network #####
                    model = Flatten()(last_output)
                    model = Dense(Fc_size)(model)
                    model = tf.nn.leaky_relu(model, alpha=0.1, name='Leaky_ReLU') 
                    model = Dropout(0.45)(model)
                    model = Dense(5, activation = 'softmax')(model)
                    model = Model(pre_trained_model.input, model) 
                      
                    
                    if Opt_sel=='Sgd':
                        opt=tf.keras.optimizers.SGD(learning_rate=Lr, momentum=0.5, nesterov=False, name="SGD")
                    if Opt_sel=='Nadam':
                        opt=tf.keras.optimizers.Nadam(learning_rate=Lr, beta_1=0.9, beta_2=0.99)
                    if Opt_sel=='Rmsprop':
                        opt=tf.keras.optimizers.RMSprop(learning_rate=Lr,rho=0.9,momentum=0.5,epsilon=1e-07,centered=False,name="RMSprop")
                                 
                                    
                    model.compile(loss="sparse_categorical_crossentropy", 
                                  optimizer=opt, 
                                  metrics=["sparse_categorical_accuracy"])
                    
                    
    #                model.summary()
                    
                    
                    import time
                    
                    def TicTocGenerator():
                        # Generator that returns time differences
                        ti = 0           # initial time
                        tf = time.time() # final time
                        while True:
                            ti = tf
                            tf = time.time()
                            yield tf-ti # returns the time difference
                    
                    TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
                    
                    # This will be the main function through which we define both tic() and toc()
                    def toc(tempBool=True):
                        # Prints the time difference yielded by generator instance TicToc
                        tempTimeInterval = next(TicToc)
                        if tempBool:
                            print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
                    
                    def tic():
                        # Records a time in TicToc, marks the beginning of a time interval
                        toc(False)
                     
                        
                    tic()
                    
                    print(modelo,capa,Opt_sel,Lr, Fc_size,FZ_point,)
                    """
                    TRAIN MODEL
                    """
                    callback = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3)
                    
                    history = model.fit(X_train,y_train,
                                        batch_size=32,
                                        epochs=Epoch_length, 
                                        callbacks=[callback],
                                        validation_data=(X_valid, y_valid))
                    
                    plot_result(history)
                    
                    "SAVE MODEL"
                    
                    model.save(modelo +"_"+"symptom" + str(count_mod)+"-"+str(Lr)+"-"+str(Fc_size)+"-"+str(FZ_point) + "-"+ Opt_sel +'.h5')                 
                    
                    
                    
                    
                    """
                    EVALUATE MODEL
                    """
                    
                    from yellowbrick.datasets import load_credit
                    from yellowbrick.classifier import confusion_matrix
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.model_selection import train_test_split as tts
                    from sklearn.metrics import confusion_matrix
                    from sklearn.metrics import accuracy_score, classification_report
                    # The ConfusionMatrix visualizer taxes a model
                    
                    
                    #CLASSIFICATION REPORT
                    test_loss, test_accuracy = model.evaluate(X_test, y_test)
                    print("Test accuracy: {}".format(test_accuracy))
                    predicciones = model.predict(X_test)
                    y_pred = np.argmax(predicciones, axis=1)
                    score = accuracy_score(y_test, y_pred)
                    import pandas as pd
                    data_pandas = pd.DataFrame()
                    data_pandas['y_pred'] = y_pred
                    data_pandas['y_test'] = y_test
                    data_pandas
                    print(classification_report(y_test, y_pred))
                    
                    """
                    PLOT CONFUSION MATRIX
                    """
                    
                    from sklearn.metrics import confusion_matrix
                    print(confusion_matrix(y_test, y_pred))
                    confmat=confusion_matrix(y_test, y_pred)
                    fig,ax=plt.subplots(figsize=(2.5,2.5))
                    ax.matshow(confmat, cmap=plt.cm.Reds, alpha=0.5)
                    for i in range(confmat.shape[0]):
                        for j in range(confmat.shape[1]):
                            ax.text(x=j, y=i,s=confmat[i,j],va='center',ha='center')
                    plt.xlabel('prediced_label')
                    plt.ylabel('true_label')
                    plt.show()
                    
                    #report_confusion_matrix = confusion_matrix(y_test, y_pred,output_dict=True)
                    
                    # Show images predicted
                    import random
                    n_images = 10
                    len_data = len(X_test)
                    ncols = 5
                    nrows = n_images//ncols 
                    # Select random numbers of Test Data
                    # fig = plt.figure( figsize=(ncols*1.5, nrows*1.5), figsize=(18,18))
                    fig, axs = plt.subplots(nrows, ncols, figsize=(12,8))
                    for i in range(n_images):
                        IMG = random.randint(0,len_data-1)
                        j=0
                        if i >= (n_images/2):
                            j = 1
                            i = i - int(n_images/2)
                        axs[j, i].imshow(X_test[IMG])
                        axs[j, i].set_title(f"P: {y_pred[IMG]} - R: {y_test[IMG]}")
                        #plt.axis('off')
                        #plt.show()
                        
                    predicciones = model.predict(X_test)
                    y_pred = np.argmax(predicciones, axis=1)
                    score = accuracy_score(y_test, y_pred)
                    print(f"Evaluate Score: {score}")
                    print(model.evaluate(X_test, y_test))
                    
                    import pandas as pd
                    from sklearn import metrics
                    print(metrics.classification_report(y_test, y_pred))
                    report = classification_report(y_test, y_pred, output_dict=True)
                    df = pd.DataFrame(report).T  # transpose to look just like the sheet above
                    df_h = pd.DataFrame(history.history).T  # transpose to look just like the sheet above
                    df.to_excel(OUTPUTH_PATH  + '\p' +"-" + modelo+"-" + str(count_mod)+"-"+str(Lr)+"-"+str(Fc_size)+"-"+str(FZ_point) + "-"+ Opt_sel +'.xlsx')
                    df_h.to_excel(OUTPUTH_PATH  + '\h' + "-" + modelo +"-" +str(count_mod)+ "-"+str(Lr)+"-"+str(Fc_size)+"-"+str(FZ_point)+"-"+ Opt_sel +'.xlsx')
                    
