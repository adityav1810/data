import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import tarfile
import requests
import re
import sys
import warnings
warnings.filterwarnings('ignore')
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
# import matplotlib.pyplot as plt
# import cv2
import pandas as pd
import pickle
from numpy.linalg import norm

# helper functions 
# load data
def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data, y_data
      
def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
            
def get_norm(weights,ord):
  norm = []
  for i in range( weights.shape[3]):
    norm.append(tf.norm(weights[:,:,:,i],ord=ord).numpy())
  norm =np.array(norm)
  return norm,np.argsort(norm),np.mean(norm)

data_path="/content/drive/MyDrive/fine-pruning/data/Lab3"

clean_data_valid_filename = data_path+"/cl/valid.h5"
clean_data_test_filename = data_path+"/cl/test.h5"

bd_valid_filename = data_path+"/bd/bd_valid.h5"
bd_test_filename = data_path+"/bd/bd_test.h5"


model_path="/content/drive/MyDrive/fine-pruning/models"
cleanModel_path = model_path+"/bd_net.h5"
cleanModel_weights_path = model_path+"/bd_weights.h5"

backdoorModel_path = model_path+"/bd_net_tmp.h5"
backdoorModel_weights_path = model_path+"/bd_weights_tmp.h5"



lr = 1e-3
epochs = 10
batch_size = 32


parser = argparse.ArgumentParser()
parser.add_argument("--layeridx", help="please enter valid layeridx [1,3,5]",type=int)
parser.add_argument("--norm", help="please enter valid norm [l1norm ,l2norm] ")

args = parser.parse_args()
print(type(args.layeridx),args.layeridx)
print(type(args.norm),args.norm)    

conv_layer_idx = args.layeridx

prune_method = args.norm

print(type(conv_layer_idx),conv_layer_idx)
print(type(prune_method),prune_method)  

def finePruning(conv_layer_idx,B_path,B_weights_path,B_prime_path, B_prime_weights_path, lr, epochs, batch_size, percentChRemovedThreshold, clean_data_valid_filename,clean_data_test_filename,poisoned_data_test_filename =None,prune_method="l1norm",verbose=False):
  if prune_method not in {'l1norm', 'l2norm', 'apoz'}:
    raise ValueError('Invalid `pruning method`')

    # define optimizer
  opt = tf.keras.optimizers.Adam(learning_rate=lr)
    # load baseline model
  B = keras.models.load_model(B_path)
  B.load_weights(B_weights_path)
    
    # define the B_prime model and initialize it with the same weights as B (initally it is the same as the baseline)
  B_prime = keras.models.load_model(B_path)
  B_prime.load_weights(B_weights_path)
    
   
  cl_x_valid, cl_y_valid = data_loader(clean_data_valid_filename)
  cl_x_valid_t, cl_y_valid_t = data_loader(clean_data_valid_filename)


  cl_x_test, cl_y_test = data_loader(clean_data_test_filename)
  bd_x_test, bd_y_test = data_loader(poisoned_data_test_filename)
    
        
    # evaluate the original model accuracy on the clean validation data
  cl_label_p_valid_orig = np.argmax(B_prime(cl_x_valid_t), axis=1)
  clean_accuracy_valid_orig = np.mean(np.equal(cl_label_p_valid_orig, cl_y_valid_t)) * 100

    # evaluate the original model accuracy on the clean test data
  cl_label_p_test_orig = np.argmax(B_prime(cl_x_test), axis=1)
  clean_accuracy_test_orig = np.mean(np.equal(cl_label_p_test_orig, cl_y_test)) * 100
  print("Clean validation accuracy before modification: {0:3.6f}".format(clean_accuracy_valid_orig))
  print("Clean test accuracy before modification: {0:3.6f}".format(clean_accuracy_test_orig))
  bd_label_p_test_orig = np.argmax(B_prime(bd_x_test), axis=1)
  asr_test_orig= np.mean(np.equal(bd_label_p_test_orig, bd_y_test)) * 100
  print("Attack success rate before modification: {0:3.6f}".format(asr_test_orig))
    
  convLayerWeights = B.layers[conv_layer_idx].get_weights()[0]
  convLayerBiases  = B.layers[conv_layer_idx].get_weights()[1]

  if prune_method == "l1norm":

      # get the activations and sort them in an increasing order excluding empty layers
      norms,allIdxToPrune,thresh = get_norm(convLayerWeights,ord = 1)

      idxToPrune = []
      for idx in allIdxToPrune :
        if norms[idx] < thresh:
          idxToPrune.append(idx)
      if verbose ==True :
        print(idxToPrune)
   
  elif prune_method == "l2norm":
      norms,allIdxToPrune,thresh = get_norm(convLayerWeights,ord = "euclidean")

      idxToPrune = []
      for idx in allIdxToPrune :
        if norms[idx] < thresh:
          idxToPrune.append(idx)
      if verbose ==True :
        print(idxToPrune)

  # elif prune_method == "apoz":
  #     layer=B_prime.layers[conv_layer_idx]
  #     norms = identify.get_apoz(B_prime, layer, bd_x_test)
  #     allIdxToPrune = np.argsort(norms)
  #     idxToPrune = identify.high_apoz(norms,method="both")

  res_shape = convLayerWeights.shape[3] + 1

  totalIters=np.zeros((res_shape))  
  totalPercentChannelsRemoved = np.zeros((res_shape))
  totalCleanAccuracyValid = np.zeros((res_shape))
  totalAttackSuccessRateValid = np.zeros((res_shape))
  totalCleanAccuracyTest = np.zeros((res_shape))
  totalAttackSuccessRateTest = np.zeros((res_shape))
  percentValidationAccuracy = []

  iter = 0
    # before the fine pruning estimate the baseline clean validation/test accuracies:
  percentChannelsRemoved = 0 # no channel has been removed yet
  totalIters[iter] = iter
  totalPercentChannelsRemoved[iter] = percentChannelsRemoved
  totalCleanAccuracyValid[iter] = clean_accuracy_valid_orig
  totalCleanAccuracyTest[iter] = clean_accuracy_test_orig
  totalAttackSuccessRateTest[iter] = asr_test_orig
  iter+=1

  for idx in allIdxToPrune:

    percentChannelsRemoved = iter / convLayerWeights.shape[3]

    if norms[idx] <thresh:
      if verbose == True:

        print("pruning idx : ",idx)


      
      
      #set weight and bias value of idx to zero
      convLayerWeights[:,:,:,idx] = 0
      convLayerBiases[idx] =  0
      B_prime.layers[conv_layer_idx].set_weights([convLayerWeights,convLayerBiases])

      #train model in updated weights 
      B_prime.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
      B_prime.fit(cl_x_valid, cl_y_valid, epochs=epochs, batch_size=batch_size,verbose=verbose)
      tf.keras.backend.clear_session()

      #evaluate model on clean validation data
      cl_label_p_valid = np.argmax(B_prime(cl_x_valid_t), axis=1)
      clean_accuracy_valid = np.mean(np.equal(cl_label_p_valid, cl_y_valid_t)) * 100

      #evalute model on clean test data
      cl_label_p_test = np.argmax(B_prime(cl_x_test), axis=1)
      clean_accuracy_test = np.mean(np.equal(cl_label_p_test, cl_y_test)) * 100

      #evaluate attack success rate
      bd_label_p_test = np.argmax(B_prime(bd_x_test), axis=1)
      asr_test = np.mean(np.equal(bd_label_p_test, bd_y_test)) * 100
      if verbose == True:
        print("Iteration = {0:3d}, channel removed = {1:3d}, percent channels removed = {2:3.6f}\nClean validation accuracy after modification: {3:3.6f}\n Clean test accuracy after modification: {4:3.6f}, attack success rate test =  {5:3.6f}".format(
                    iter, idx, percentChannelsRemoved * 100, clean_accuracy_valid, clean_accuracy_test, asr_test))

      #save metrics
      totalPercentChannelsRemoved[iter] = percentChannelsRemoved
      totalCleanAccuracyValid[iter] = clean_accuracy_valid
      totalCleanAccuracyTest[iter] = clean_accuracy_test
      totalAttackSuccessRateTest[iter] = asr_test
      totalIters[iter] = iter
      iter = iter + 1

    else: # element not pruned
      if verbose == True:
        print("Iteration = {0:3d}\nClean validation accuracy after modification: {1:3.6f}\n Clean test accuracy after modification: {2:3.6f}, attack success rate test =  {3:3.6f}".format(
                    iter, clean_accuracy_valid, clean_accuracy_test, asr_test))
      
      totalPercentChannelsRemoved[iter] = percentChannelsRemoved
      totalCleanAccuracyValid[iter] = clean_accuracy_valid
      totalCleanAccuracyTest[iter] = clean_accuracy_test
      totalAttackSuccessRateTest[iter] = asr_test
      totalIters[iter] = iter
      iter = iter + 1
      
    
  B_prime.save(B_prime_path)
  B_prime.save_weights(B_prime_weights_path)
  return (totalPercentChannelsRemoved,totalCleanAccuracyValid,totalCleanAccuracyTest,totalAttackSuccessRateTest)





(totalIters,totalPercentChannelsRemoved,totalCleanAccuracyValid,totalCleanAccuracyTest,totalAttackSuccessRateTest) = finePruning(conv_layer_idx,cleanModel_path,cleanModel_weights_path,
                                                                                                                    backdoorModel_path,backdoorModel_weights_path,
                                                                                                                    lr, epochs, batch_size, 100, 
                                                                                                                    clean_data_valid_filename,clean_data_test_filename, 
                                                                                                                    bd_test_filename,
                                                                                                                    prune_method=prune_method,
                                                                                                                    verbose=True)


df = pd.DataFrame({'totalIters':totalIters,'totalPercentChannelsRemoved':totalPercentChannelsRemoved,'totalCleanAccuracyValid':totalCleanAccuracyValid,'totalCleanAccuracyTest':totalCleanAccuracyTest,'totalAttackSuccessRateTest':totalAttackSuccessRateTest})
df.to_csv("/content/results-"+prune_method+"-convlayer-"+str(conv_layer_idx)+".csv", index=False)



