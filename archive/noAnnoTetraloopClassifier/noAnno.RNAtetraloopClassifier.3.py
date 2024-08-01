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
from RNAArrayUtils import plotMulticlassConfusionMatrix
from RNAArrayUtils import mergeTrainTestUniquify

LOAD_WEIGHTS=0
EPOCHS = 2 # was 20
BATCH_SIZE = 8
NUM_CLASSES=3  #3 # max is 45
#mpl.rcParams['figure.figsize'] = (100,100)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


#preparing dataset old way:
#x_train, y_train, x_test, y_test = prepareDatasetOldWay()
#print("shapes of  x_train.shape, y_train.shape, x_test.shape, y_test.shape : ", x_train.shape, y_train.shape, x_test.shape, y_test.shape )
# with old way, got accuracy of .9925 after 1 epoch:
#Epoch 1/20
#189195/189205 [============================>.] - ETA: 0s - loss: 1.1014 - tp: 0.0000e+00 - fp: 4.0000 - tn: 3027116.0000 - fn: 1513560.0000 - accuracy: 0.9925 - precision: 0.0000e+00 - recall: 0.0000e+00 - auc: 0.9940 - prc: 0.9889  I

#preparing dataset new way:
x_train, y_train, x_test, y_test = mergeTrainTestUniquify() 
print("after mergeTrainTestUniquify: shapes of  x_train.shape, y_train.shape, x_test.shape, y_test.shape : ", x_train.shape, y_train.shape, x_test.shape, y_test.shape )
# with mergeTrainTestUniquify:



# Test whether GPU is being used:
if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    #{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

if (NUM_CLASSES==45):
    y_train_oh = to_categorical(y_train, num_classes=NUM_CLASSES)  # Make labels into one-hot encode. 
    y_test_oh = to_categorical(y_test, num_classes=NUM_CLASSES)  # Make labels into one-hot encode. 
elif (NUM_CLASSES==3 ):
    y_train = np.where((y_train >(NUM_CLASSES-2) ), (NUM_CLASSES-1)  ,  y_train)
    # 0 is non-tetraloop, 1 is GNRA, 2 is non-GNRA tetraloop
    y_train_oh = to_categorical(y_train, num_classes=NUM_CLASSES)  # Make labels into one-hot encode. 
    y_test = np.where((y_test > (NUM_CLASSES-2)  ),(NUM_CLASSES-1)  ,y_test)
    y_test_oh = to_categorical(y_test, num_classes=NUM_CLASSES)  # Make labels into one-hot encode. 
elif (NUM_CLASSES==2 ):
    y_train = np.where((y_train > (NUM_CLASSES-2) ),(NUM_CLASSES-1) ,  y_train)
    # 0 is non-tetraloop, 1 is GNRA, 2 is non-GNRA tetraloop
    y_train_oh = to_categorical(y_train, num_classes=NUM_CLASSES)  # Make labels into one-hot encode. 
    y_test = np.where((y_test > (NUM_CLASSES-2)),(NUM_CLASSES-1)  ,y_test)
    y_test_oh = to_categorical(y_test, num_classes=NUM_CLASSES)  # Make labels into one-hot encode. 

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
    Dense(NUM_CLASSES, activation='softmax', bias_initializer=output_bias, kernel_initializer='glorot_uniform')  #Softmax for multiclass classification problem. Glorot for softmax.
    ])
     
  model.compile(
    # Note that the learning rate is 1e-4 which yields higher accuracy. 
    optimizer=keras.optimizers.Ftrl    (learning_rate=1e-4), #after updating to M1 GPU version of conda, had trouble with Adam optimizer
    #optimizer=keras.optimizers.Adam(learning_rate=1e-4), #optimizer=RMSprop(lr=0.001),
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
    plt.savefig("./convergence"+".tiff")
    plt.show()
    plt.clf() # Close the plt 

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
print("d_class_weights = ", d_class_weights )
d_class_weights[1] = d_class_weights[1]*1
d_class_weights[2] = d_class_weights[2]*1
print("d_class_weights = ", d_class_weights )

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


def plot_base_pair_frequencies(tetraloopClass,basePairPosition1, basePairPosition2, basePairStats,singleResidueStats):
    residueTypes = ["G","C","U","A"]
    font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 22}

    mpl.rc('font', **font)
    fig, ax = plt.subplots()
    im = ax.imshow(basePairStats)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(residueTypes)), labels=residueTypes)
    ax.set_yticks(np.arange(len(residueTypes)), labels=residueTypes)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(residueTypes)):
        for j in range(len(residueTypes)):
            text = ax.text(j, i, basePairStats[i, j],
                           ha="center", va="center", color="w")
    titleString = "Frequency of base pair at positions "+str(basePairPosition1)+", "+str(basePairPosition2)+" for tetraloop class "+str(h)
    ax.set_title(titleString)                            
    fig.tight_layout()
    plt.rcParams["figure.figsize"] = (2 , 2 )
    plt.savefig("./basePairFreq_"+str(basePairPosition1)+"_"+str(basePairPosition2)+"_"+str(h)+".tiff")
    #plt.show()
    plt.clf()

def plot_single_residue_frequencies(tetraloopClass,singleResidueStats):
    residueTypes = ["G","C","U","A"]
    font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 18}

    mpl.rc('font', **font)

    # create data
    x = np.arange(0,8)
    residueTypes=['G', 'C', 'U', 'A']
    y = np.zeros((4,8)) # position, residue type: frequency
    
    for i in range(4):
        y[i] = singleResidueStats[:,i]      
        # plot bars in stack manner
        print ("x = ",x)
        print ("y[",i,"] ", y[i])
    y=y/np.max(y)
    #plt.ylim([0, 8620])
    plt.bar(x, y[0]+y[1]+y[2]+y[3], color='g') #G
    plt.bar(x, y[1]+y[2]+y[3] , color='c') #C
    plt.bar(x,y[2]+y[3] , color='b') #U
    plt.bar(x, y[3], color='r') #A
    plt.xlabel("residue position")
    plt.xticks(np.arange(0,8),[0,1,2,3,4,5,6,7])
    plt.ylabel("frequency" )
    plt.legend(residueTypes, loc=(1.01,0.3)          ) #und 2", "Round 3", "Round 4"])
    plt.title("Frequency of residue type by position")
    plt.savefig("./singleResidueFreq_class"+str(h)+".tiff",bbox_inches="tight")
    #plt.show()
    plt.clf() # Close the plt 

np.set_printoptions(suppress=True) # Suppress scientific notation. Makes numbers more readable.

def computeFrequencies(tetraloopClass,basePairPosition1, basePairPosition2, basePairStats, singleResidueStats):
    classVector = np.where(y_train[:] == h, 1, 0)
    hasGAtPos2  = np.where(x_train[:,2,0 ] == 1, 1, 0)
    hasAAtPos5  = np.where(x_train[:,5,3 ] == 1, 1, 0)
    includeVector = np.multiply( np.multiply(classVector,hasGAtPos2),hasAAtPos5)
    for i in range(4):
        for position in range (8):
            singleResidueStats[position,i] = ((np.dot(np.multiply(includeVector,x_train[:,position , i]),x_train[:,position , i])))
        for j in range(4):
            numOfBasePairsOfGivenTypeAtGivenPosition = ((np.dot(np.multiply(includeVector,x_train[:,basePairPosition1 , i]),x_train[:,basePairPosition2 , j])))
            basePairStats[i,j]=numOfBasePairsOfGivenTypeAtGivenPosition
    print("for tetraloop class: ",tetraloopClass )
    print(basePairStats)
    print ("single residue stats considering two positions:")
    print ((singleResidueStats))
    for i in range(8):
        print (np.sum(singleResidueStats[i,:]))

for h in np.arange(1,2):
    singleResidueStats = np.zeros((8,4)) # position, residue type: frequency
    print("check 1.4")
    basePairStats = np.zeros(( 4, 4 )).astype(int)
    basePairPosition1=0
    basePairPosition2=7
    computeFrequencies(h,basePairPosition1,basePairPosition2,basePairStats,singleResidueStats)
    plot_base_pair_frequencies(h,basePairPosition1, basePairPosition2,basePairStats,singleResidueStats) 
    plot_single_residue_frequencies(h,singleResidueStats)
    #for basePairPosition1 in range (8):
    #    for basePairPosition2 in np.arange (basePairPosition1+1,8):
    #        computeFrequencies(h,basePairPosition1,basePairPosition2,basePairStats,singleResidueStats)
    #        plot_base_pair_frequencies(h,basePairPosition1, basePairPosition2,basePairStats,singleResidueStats) 
    





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
weightedModel = make_model()
#weightedModel.load_weights(initial_weights)


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

if (LOAD_WEIGHTS):
    print("loading weights:")
    weightedModel.load_weights(latest)
    print("done.")



# model.fit trains the model for a fixed number of epochs (iterations on a dataset).
#add checkpoints:
weighted_history = weightedModel.fit(
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
train_predictions_weighted = weightedModel.predict(x_train, batch_size=BATCH_SIZE)
test_predictions_weighted = weightedModel.predict(x_test, batch_size=BATCH_SIZE)
plotMulticlassConfusionMatrix(y_test,np.argmax(test_predictions_weighted, axis=1))

# Results of test data set. 
# model.evaluate returns the loss value & metrics values for the model in test mode (the model is already trained).
weighted_results = weightedModel.evaluate(x_test, y_test_oh,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weightedModel.metrics_names, weighted_results):
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




