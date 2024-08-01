import matplotlib.pyplot as plt
import numpy as np
import os

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn import preprocessing
import tensorflow as tf
import math
from numpy import asarray
from numpy import save
from numpy import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D, Conv2D, MaxPooling2D, Dropout, Flatten, Input, MaxPooling1D
from sklearn.preprocessing import StandardScaler
import os
import tempfile
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
import matplotlib as mpl
import sklearn

EPOCHS = 3 # was 20
BATCH_SIZE = 8
NUM_CLASSES=3 # max is 45
mpl.rcParams['figure.figsize'] = (15, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Load dataset.
# Please change this to the correspond directory. 
# Datasets are saved in the shared Google drive: deep-learning-RNA-structure-prediction/Chih-Fan/TetraloopsDatasets
# In case you are not the owner of this folder, you have to make a copy (either the entire folder or files) to your own drive. 
dict_train = np.load('.//TetraloopsDatasets/noAnno_train_array.npz')  
x_train = np.stack(dict_train['arr_0'], axis=0)
y_flt_train = np.load('.//TetraloopsDatasets/noAnno_train_labels.npy')

dict_test = np.load('.//TetraloopsDatasets/noAnno_test_array.npz')
x_test = np.stack(dict_test['arr_0'], axis=0)
y_flt_test = np.load('.//TetraloopsDatasets/noAnno_test_labels.npy')

if (NUM_CLASSES==45):
    y_train = y_flt_train.astype(int)
    y_train_oh = to_categorical(y_train, num_classes=45)  # Make labels into one-hot encode. 
y_test = y_flt_test.astype(int)
y_test_oh = to_categorical(y_test, num_classes=45)  # Make labels into one-hot encode. 

print('Training features shape:', x_train.shape)
print('Test features shape:', x_test.shape)
print('Training labels shape:', y_train_oh.shape)
print('Test labels shape:', y_test_oh.shape)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

print('Training labels shape:', y_train_oh.shape)
print('Test labels shape:', y_test_oh.shape)

print('Training features shape:', x_train.shape)
print('Test features shape:', x_test.shape)

def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  
  model = Sequential([
    # The input shape is 8x4
    # This is the first convolution
    #Conv1D(64, 3, strides=1, activation='relu',padding='same', #4(possible base)^3(kernal size)=64
    #             input_shape=(8, 4),
    #             kernel_initializer='he_normal',  
    #             bias_initializer='zeros'),
    #MaxPooling1D(pool_size=2, strides=2),
    #Dropout(0.2),
    #Conv1D(32, 3, strides=1, activation='relu',padding='same', 
    #             kernel_initializer='he_normal',
    #             bias_initializer='zeros'),
    #MaxPooling1D(pool_size=2, strides=2),    
    #Dropout(0.2),
    Flatten(),
    #  neuron hidden layer
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    Dense(128, activation='relu', bias_initializer=output_bias, kernel_initializer='he_normal'),  #8/2(maxpooling)=4, 32*4 = 128
    #Dropout(0.2),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for class ('not GNRA') and 1 for the other ('GNRA')
    Dense(45, activation='softmax', bias_initializer=output_bias, kernel_initializer='glorot_uniform')  #Softmax for multiclass classification problem. Glorot for softmax.
    ])
     
  model.compile(
    # Note that the learning rate is 1e-4 which yields higher accuracy. 
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), #optimizer=RMSprop(lr=0.001),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=METRICS)  # Accuracy for classification problems. 
  
  return model


# Plot loss and accuracy.
def plot_metrics(history):
  metrics = ['loss', 'accuracy']
  plt.rcParams['font.size'] = '16'
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(1,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_' + metric],
             color=colors[0], linestyle="--", label='Test')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'accuracy':
      plt.ylim([0,1])
    else:
      plt.ylim([0,1])

    plt.legend();

# Plot ROC curve.
def plot_roc(name, labels, predictions, **kwargs):
  fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)  # fpr=false positive rate = fp/(fp+tn), tpr=true positive rate = tp/(tp+fn)

  plt.plot(100*fpr, 100*tpr, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives rate = fp/(fp+tn)')
  plt.ylabel('True positives rate = tp/(tp+fn)')
  plt.xlim([-5,100])
  plt.ylim([0,100])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

# Plot precision-recall curve (for binary problem)
def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall  - TP / (TP + FP)')
    plt.ylabel('Precision - TP / (TP + FN)')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

# retrain with class weights
# The sum of the weights of all examples stays the same.
y_integers = np.argmax(y_train_oh, axis=1)
class_weights = np.round(class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers), 2)
d_class_weights = dict(enumerate(class_weights))


print("check 1.")
print((x_train[:, 7, 3]))
#print(
h=1
print("check 1.3")
"""
for h in range(45):
    classVector = np.where(y_train[:] == h, 1, 0)
    basePairStats = np.zeros(( 4, 4 ))
    for i in range(4):
        for j in range(4):
            numOfBasePairsOfGivenTypeAtGivenPosition = ((np.dot(np.multiply(classVector,x_train[:, 7, i]),x_train[:, 0, j])))
            basePairStats[i,j]=numOfBasePairsOfGivenTypeAtGivenPosition
    print("for tetraloop class: ",h)
    print(basePairStats)
"""

"""
for h in range(45):
    classVector = np.where(y_train[:] == h, 1, 0)
    basePairStats = np.zeros(( 4, 4 )).astype(int)
    for i in range(4):
        for j in range(4):
            numOfBasePairsOfGivenTypeAtGivenPosition = ((np.dot(np.multiply(classVector,x_train[:, 4, i]),x_train[:, 3, j])))
            basePairStats[i,j]=numOfBasePairsOfGivenTypeAtGivenPosition
    print("for tetraloop class: ",h)
    print(basePairStats)
"""




#print(classVector)
print(x_train[0][7][3])
print(x_train[1][7][3])

#for h in range (45): 
#    basePairStats = np.zeros(( 4, 4 ))
#    for i in range (len(y_train)): 
#        if (y_train[i] == h) :
#            basePairStats[np.argmax(x_train[i][7])][np.argmax(x_train[i][0])]=basePairStats[np.argmax(x_train[i][7])][np.argmax(x_train[i][0])]+1
#    print ("base pair stats for tetraloop type ",h)
#    print(basePairStats)

print("check 2")
#print(basePairStats)
print("check 3")
print(d_class_weights)

# Train the CNN model on the training set and evaluate by test set.
weighted_model = make_model()
#weighted_model.load_weights(initial_weights)


# Create a callback that saves the model's weights
# to save models:
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
os.listdir(checkpoint_dir)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print("check 1")
latest
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

LOAD_WEIGHTS=0
if (LOAD_WEIGHTS):
    print("loading weights:")
    weighted_model.load_weights(latest)
    print("done.")



# model.fit trains the model for a fixed number of epochs (iterations on a dataset).
#add checkpoints:
weighted_history = weighted_model.fit(
    x_train,
    y_train_oh, 
    validation_data = (x_test, y_test_oh),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[cp_callback],
    # The class weights go here
    class_weight=d_class_weights)



plot_metrics(weighted_history)

# Model.predict gives the predicted output. 
train_predictions_weighted = weighted_model.predict(x_train, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(x_test, batch_size=BATCH_SIZE)

# Results of test data set. 
# model.evaluate returns the loss value & metrics values for the model in test mode (the model is already trained).
weighted_results = weighted_model.evaluate(x_test, y_test_oh,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
  print(name, ': ', value)
print()

#this has not been defined
#plot_cm(y_test_oh, test_predictions_weighted)

#these are not available for multiclass results_:
#plot_roc("Train Weighted", y_train, train_predictions_weighted, color=colors[1])
#plot_roc("Test Weighted", y_test, test_predictions_weighted, color=colors[1], linestyle='--')
#plt.legend(loc='lower right');
#plot_prc("Train Weighted", y_train, train_predictions_weighted, color=colors[1])
#plot_prc("Test Weighted", y_test, test_predictions_weighted, color=colors[1], linestyle='--')
#plt.legend(loc='lower right');


