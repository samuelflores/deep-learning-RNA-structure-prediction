import matplotlib.pyplot as plt
import numpy as np
import os

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
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

EPOCHS = 20 # was 20
BATCH_SIZE = 8
NUM_CLASSES=3 # max is 45
mpl.rcParams['figure.figsize'] = (15, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
print ("tf.config.threading.get_intra_op_parallelism_threads = ",tf.config.threading.get_intra_op_parallelism_threads())
tf.config.threading.set_intra_op_parallelism_threads(30)
print ("tf.config.threading.get_intra_op_parallelism_threads = ",tf.config.threading.get_intra_op_parallelism_threads())
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
y_test = y_flt_test.astype(int) # Be careful: astype(int) rounds using trunc.

y_train = y_flt_train.astype(int)

print ("check 11 np.shape(x_train) = ",np.shape(x_train))
print ("check 11 np.shape(x_test ) = ",np.shape(x_test ))
x_train =np.flip( x_train,axis=1) # For some reason the sequences were all in reverse order. Flip along residue-number axis to make it more cognitively reasonable. y_train should NOT be flipped.
x_test  =np.flip( x_test ,axis=1)




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

def shuffle_along_axis(x,y, axis): # This does not work yet
    idx = np.random.rand(*x.shape).argsort(axis=axis)
    print (idx)
    print ("shape of idx = ",idx.shape)
    x =    np.take_along_axis(x,idx,axis=axis)
    y =    np.take_along_axis(y,idx[:,0,0],axis=axis)

def dataStats ():                                    
    print("Basic stats about our train and test sets:")
    print('Training features shape:', x_train.shape)
    print('Test features shape:', x_test.shape)
    print('Training labels shape:', y_train_oh.shape)
    print('Test labels shape:', y_test_oh.shape)
    for i in range (NUM_CLASSES):
        print ('Train set data points in category ',i," : ", sum(y_train_oh[:,i])," , ",sum(y_train_oh[:,i])*100/len(y_train_oh),"%")
        print ('Test set data points in category  ',i," : ", sum(y_test_oh[:,i]) ," , ",sum(y_test_oh[:,i]) *100/len(y_test_oh),"%")
        print ("Train set average position of category ",i," ",np.dot( np.arange(0,len(y_train_oh)), y_train_oh[:,i])/ sum(y_train_oh[:,i]))
        print ("Test  set average position of category ",i," ",np.dot( np.arange(0,len(y_test_oh)) , y_test_oh[:,i]) / sum(y_test_oh[:,i]))
    print("randomizing order along axis 0:")    
    #shuffle_along_axis(x_train,y_train_oh,0)
    #shuffle_along_axis(x_test,y_test_oh,0)
    #print("after randomizing order along axis 0:")    
    for i in range (NUM_CLASSES):
        print ('Train set data points in category ',i," : ", sum(y_train_oh[:,i])," , ",sum(y_train_oh[:,i])*100/len(y_train_oh),"%")
        print ('Test set data points in category  ',i," : ", sum(y_test_oh[:,i]) ," , ",sum(y_test_oh[:,i]) *100/len(y_test_oh),"%")
        print ("Train set average position of category ",i," ",np.dot( np.arange(0,len(y_train_oh)), y_train_oh[:,i])/ sum(y_train_oh[:,i]))
        print ("Test  set average position of category ",i," ",np.dot( np.arange(0,len(y_test_oh)) , y_test_oh[:,i]) / sum(y_test_oh[:,i]))

dataStats()

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

#print('Training labels shape:', y_train_oh.shape)
#print('Test labels shape:', y_test_oh.shape)
#print('Training features shape:', x_train.shape)
#print('Test features shape:', x_test.shape)

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
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), #optimizer=RMSprop(lr=0.001),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=METRICS)  # Accuracy for classification problems. 
  
  return model


# Plot loss and accuracy.
def plot_metrics(history):
    metrics = ['loss', 'accuracy']
    for n, metric in enumerate(metrics):
        plt.rcParams['font.size'] = '16'
        name = metric.replace("_"," ").capitalize()
        #plt.subplot(1,2,n+1)
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
        plt.savefig("./convergence"+"."+metric+".tiff")
        #plt.show()
        plt.clf() # Close the plt

#Plot multiclass confusion matrix:
def plotMulticlassConfusionMatrix(goldStandardResult, testResult ):
    multiLabelConfusionMatrix = multilabel_confusion_matrix(goldStandardResult, testResult ) 
    print("multiclass confusion matrix:")
    print(multiLabelConfusionMatrix)
    fig = plt.figure()
    fig, axs = plt.subplots(1,NUM_CLASSES)
    for i in range(NUM_CLASSES): # here generate subplots:
        print("working on subplot : ",i)
        if (NUM_CLASSES == 3):
            axs[0].set_title("Non-tetraloop")
            axs[1].set_title("GNRA tetraloop")
            axs[2].set_title("Non-GNRA tetraloop")
        axs[i].imshow( multiLabelConfusionMatrix[i] , cmap = 'jet'  )
        axs[i].set_xticks(np.arange(2),labels=["Test negative", "Test positive"])
        axs[i].set_yticks(np.arange(2),labels=["", ""]) # Since this is the left axis, all but the first can have blank labels.
        axs[0].set_yticks(np.arange(2),labels=["GS negative", "GS positive"]) # This is the first, and only one that needs y axis labels.
        # Loop over data dimensions and create text annotations.
        for m in range((2)):
            for n in range((2)):
                myText = ""
                if ((m == 0) and (n==0)) :
                    myText = "TN "
                elif ((m == 1) and (n==0)) :
                    myText = "FN "
                elif ((m == 0) and (n==1)) :
                    myText = "FP "
                else  :
                    myText = "TP "
                myText = myText + str(multiLabelConfusionMatrix[i][m, n])    
                text = axs[i].text(n, m, myText,
                         ha="center", va="center", color="w")
        #im.imsave("confusionMatrix."+str(i)+".tiff")
        print()
    #plt.imshow( multiLabelConfusionMatrix , cmap = 'jet'  )
    #plt.show()
    fig.savefig("multiclassConfusionMatrix.tiff")
    plt.clf()


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
    print("before loading weights:")
    #test_predictions_weighted_before_load = weighted_model.predict(x_test, batch_size=BATCH_SIZE) #
    #print("prediction computed.   ")
    #print("Producing plot:        ")
    #plotMulticlassConfusionMatrix(y_test,(np.argmax(test_predictions_weighted_before_load, axis=1))) # These results were also crap, as one should expect.         
    print("loading weights:")
    weighted_model.load_weights(latest)
    print("weights loaded.")
    print("compute prediction:")
    test_predictions_weighted_before_fit =np.argmax( weighted_model.predict(x_test, batch_size=BATCH_SIZE), axis=1) # weighted_model.predict returns 3 floats, we convert to an int representing whichever of the NUM_CLASSES was predicted
    print("prediction computed.   ")
    #Test 
    plotMulticlassConfusionMatrix(y_test,((test_predictions_weighted_before_fit))) # Do not use astype(int). That rounds using trunc.
    for i in range (1000): # could also be len(y_test)
        print ("predicted, gold standard result for test data point i =",i, " : ", test_predictions_weighted_before_fit[i], y_test[i]) 
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
plotMulticlassConfusionMatrix(y_test,np.argmax(test_predictions_weighted, axis=1))
    
#plotMulticlassConfusionMatrix(y_test,(np.argmax(test_predictions_weighted_before_load, axis=1))) # These results were also crap, as one should expect.         

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


