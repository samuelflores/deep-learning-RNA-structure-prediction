import numpy as np

# This takes an array (arrayToExpand) and duplicates each row i arrayToExpand[i,columnNumberWithExpansionCount] times
def expandRepeats (arrayToExpand, columnNumberWithExpansionCount):
    print ("shape before inserting  repeats: ", arrayToExpand.shape)
    total_rows_arrayToExpand_expanded = np.sum(arrayToExpand[:,columnNumberWithExpansionCount].astype(int))
    print ("total_rows_arrayToExpand_expanded = ",total_rows_arrayToExpand_expanded," . This is the length of the fully expanded output array."  )
    arrayToExpand_expanded =   np.empty_like(arrayToExpand)
    arrayToExpand_expanded=np.delete(arrayToExpand_expanded, np.s_[1:arrayToExpand_expanded.shape[0]],axis=0)   # delete all but one
    for j in range(arrayToExpand_expanded.shape[1]):
        if arrayToExpand_expanded[0, j] is str:
            arrayToExpand_expanded[0, j] = None
        else:
            arrayToExpand_expanded[0, j] = 0 
    arrayToExpand_expanded = np.repeat(arrayToExpand_expanded[0],repeats=( total_rows_arrayToExpand_expanded  ),axis = 0)
    arrayToExpand_expanded = arrayToExpand_expanded.reshape(total_rows_arrayToExpand_expanded ,-1) 
    print ("shape of arrayToExpand_expanded: " ,arrayToExpand_expanded.shape)

    insertionIndex = 0 
    for rowBeingExpanded in range(arrayToExpand.shape[0]):
        if (int(arrayToExpand[rowBeingExpanded,columnNumberWithExpansionCount]) > 0 ):
            #print("about to do np.repeat ",arrayToExpand[rowBeingExpanded,columnNumberWithExpansionCount], " times. The row to be repeated is:")
            #print(arrayToExpand[rowBeingExpanded])
            temp_repeats = np.repeat(arrayToExpand[rowBeingExpanded],repeats=(arrayToExpand[rowBeingExpanded,columnNumberWithExpansionCount].astype(int)), axis=0 )
            #print ("shape of arrayToExpand_expanded: " ,arrayToExpand_expanded.shape)
            temp_repeats = temp_repeats.reshape((int(arrayToExpand[rowBeingExpanded,columnNumberWithExpansionCount])),37, order='F' ) # with default C-order, this generated across-striping rather than down-striping. Now trying F-ordingering, which should write with first index changing fastest. So apparently np.repeat is compatible with F-order.
            #print ("shape of temp_repeats: " , temp_repeats.shape)
            #print ("check 4.5 temp_repeats: ")
            #for j in range(temp_repeats.shape[0]) : print ("temp_repeats[ ",j,"] =",temp_repeats[j])
            #print ("shape of temp_repeats: " , temp_repeats.shape)
            #print("about to do insert temp_repeats of shape ",temp_repeats.shape ," into arrayToExpand_expanded[ ",insertionIndex , ":",insertionIndex+int(arrayToExpand[rowBeingExpanded,columnNumberWithExpansionCount]) ,"] of ",arrayToExpand_expanded.shape[0])
            arrayToExpand_expanded[insertionIndex:insertionIndex+int(arrayToExpand[rowBeingExpanded,columnNumberWithExpansionCount])] = temp_repeats 

            #for j in range(insertionIndex,insertionIndex+int(arrayToExpand[rowBeingExpanded,columnNumberWithExpansionCount])): print ("check 5 arrayToExpand_expanded[",j,",0] = ", arrayToExpand_expanded[j,0].astype(int))
            insertionIndex = insertionIndex + int(arrayToExpand[rowBeingExpanded,columnNumberWithExpansionCount])
        else:
            print("expansion count should never be less than unity! arrayToExpand[",rowBeingExpanded,"] = ",arrayToExpand[rowBeingExpanded]) 
            print(" arrayToExpand[",rowBeingExpanded,",",columnNumberWithExpansionCount,"] = ",arrayToExpand[rowBeingExpanded,columnNumberWithExpansionCount]) 
            sys.exit("Fatal error.")
    print ("shape after inserting repeats: ", arrayToExpand_expanded.shape)
    arrayToExpand = arrayToExpand_expanded 
    return arrayToExpand


# This procedure may appear a bit long. However I believe it is necessary. This is what it does:
# 1. merges the pre-defined train & test sets into a single dataset.
# 2. Figures out which data points (feature+label) are redundant and compresses them into unique rows, but keeps track of how many duplicates originally existed.
# 3. Figures out which data points are redundant at the level of FEATURE only. There are some feature vectors which appear with more than one label.
# 4. Sends all data points of a single FEATURE vector to either test or train. Ensures that no FEATURE vector appears in both test and train.
# 5. Expands the (feature+label) rows according to the saved counts. This restores the redundancy which contains useful training info. (consider NOT expanding but using the count number!)

def mergeTrainTestUniquify():
    dict_train = np.load('.//TetraloopsDatasets/noAnno_train_array.npz')  # This is a dictionary, 1513639 entries
    dict_test = np.load('.//TetraloopsDatasets/noAnno_test_array.npz')

    x_train_original = np.stack(dict_train['arr_0'], axis=0) # this is now an array of shape (1513639,8,4)
    x_test_original  = np.stack(dict_test['arr_0'], axis=0)
    xTestTrainMerged = np.concatenate((x_test_original, x_train_original),  axis=0)
    xTestTrainMerged = xTestTrainMerged.astype(int)
    xTestTrainMerged = np.flip(xTestTrainMerged ,axis=1) # For some reason the sequences were all in reverse order. Flip to make it more cognitively reasonable.
    print("confirm shape of xTestTrainMerged is now (2270484, ) : ", xTestTrainMerged.shape)

    y_flt_train = np.load('.//TetraloopsDatasets/noAnno_train_labels.npy')
    y_flt_test = np.load('.//TetraloopsDatasets/noAnno_test_labels.npy')
    y_train = y_flt_train.astype(int)
    y_test = y_flt_test.astype(int)
    yTestTrainMerged = np.concatenate((y_test,y_train),axis=0)
    #yTestTrainMerged = yTestTrainMerged.reshape(-1, 1, 1, 1)
    print("confirm shape of yTestTrainMerged is now (2270484, ) : ", yTestTrainMerged.shape)


    # Reshape the data and labels arrays to have shape (n, 1) for concatenation
    xTestTrainMergedReshaped = np.reshape(xTestTrainMerged , (xTestTrainMerged.shape[0], -1))
    print("Confirm that xTestTrainMergedReshaped now has shape (2270484, 32) : ", xTestTrainMergedReshaped.shape)
     
   
    # Concatenate the data and labels arrays along the last axis
    yTestTrainMergedReshaped = yTestTrainMerged.reshape(-1,1)
    xyTestTrainMergedReshaped = np.concatenate((xTestTrainMergedReshaped, yTestTrainMergedReshaped ), axis=-1)
    print("Confirm that xyTestTrainMergedReshaped now has shape (2270484, 33) : ", xyTestTrainMergedReshaped.shape)
    
    # Convert the concatenated int array to a string array
    xyTestTrainMergedReshapedAsString = xyTestTrainMergedReshaped.astype(str) #This operation is expensive

    #for i in range(xyTestTrainMergedReshapedAsString.shape[0]): print ("xyTestTrainMergedReshapedAsString[",i,",0] = ", xyTestTrainMergedReshapedAsString[i,0].astype(int))
    print("Confirm that xyTestTrainMergedReshapedAsString now has shape (2270484, 33) : ", xyTestTrainMergedReshapedAsString.shape)
    
    # Flatten the string array into a 2D array with shape (n, m) for uniqueness checking
    #data_str = xyTestTrainMergedReshapedAsString.reshape(-1, xyTestTrainMerged.shape[-1]) #should have shape (2270484,33)
    data_str = xyTestTrainMergedReshapedAsString   

    flat_data = np.array([' '.join(row) for row in data_str])
    print("done with join")
    # count the unique rows in the flattened array
    #note that in the case of a single sequence appearing multiple times, it may fall into more than one class. Probably not ideal if the same sequence appears in test and train, but with different labels.
    unique_feature_and_label_rows, unique_counts = np.unique(flat_data, return_counts=True)
    print("Now unique_feature_and_label_rows should have shape (n,) where n < 2270484 : ", unique_feature_and_label_rows.shape)   
    # convert the unique rows back to a (2270484, 8, 4, 1) array
    # first, convert the single string back to a vector of 33 strings:
    flat_data_in_33_array = np.array([ row_str.split() for row_str in unique_feature_and_label_rows])
    print("Now flat_data_in_33_array should have shape (n,33) where n < 2270484 : ", flat_data_in_33_array.shape)   
    # now column 33 will contain the count for each row in unique_feature_and_label_rows
    flat_data_in_34_array = np.concatenate((flat_data_in_33_array,unique_counts.reshape(-1,1)), axis=1)
    print("Now flat_data_in_34_array should have shape (n,34) where n < 2270484 : ", flat_data_in_34_array.shape)   
    # now we will join columns 0-31, the feature columns. We will then sort and number the unique features.
    columns0_31 = flat_data_in_34_array[:,0:32]
    print("Now columns0_31 should have shape (n,32) where n < 2270484 : ", columns0_31.shape)   
    features_joined = np.array([' '.join(row) for row in columns0_31]) 
    print("Now features_joined should have shape (n, ) where n < 2270484 : ", features_joined.shape)   
    #flat_data_in_35_array contains: columns 0-31: features. 32: labels. 33:count of feature+label. 34: features as single string (for sorting)
    flat_data_in_35_array = np.concatenate((flat_data_in_34_array,features_joined.reshape(-1,1)), axis=1)
    #flat_data_in_35_array will be sorted on column 34:
    flat_data_in_35_array [ flat_data_in_35_array [:,34].argsort() ]
    #flat_data_in_36_array contains: columns 0-31: features. 32: labels. 33:count of feature+label. 34: features as single string (for sorting). 35: index for grouping unique feature vectors:
    flat_data_in_36_array =  np.concatenate((flat_data_in_35_array,np.zeros((flat_data_in_35_array.shape[0],1),dtype=int)),axis=1)
    #for i in range(flat_data_in_36_array.shape[0]): print ("check 2 flat_data_in_36_array[",i,",0] = ", flat_data_in_36_array[i,0].astype(int))
    lastFeatureString = ""
    index =int(-1)
    for i in range (flat_data_in_36_array.shape[0]):
        #print (flat_data_in_36_array[i,34]," label: ", flat_data_in_36_array[i,32] , " feature-label count: ",  flat_data_in_36_array[i,33])
        if (flat_data_in_36_array[i,34] != lastFeatureString):
            print(" flat_data_in_36_array[,",i,",34] = ",flat_data_in_36_array[i,34])
            index = index + 1
        flat_data_in_36_array[i,35] = index
        #print (flat_data_in_36_array[i,34]," label: ", flat_data_in_36_array[i,32] , " feature-label count: ",  flat_data_in_36_array[i,33], " feature index: ", index)
        lastFeatureString = flat_data_in_36_array[i,34]
    print("number of unique feature vectors: ", index) # this is giving me 37991, expected 65536 or nearly that many.
    print("Now flat_data_in_36_array should have shape (n,36) where n < 2270484 : ", flat_data_in_36_array.shape)   
   
    # create an array of indices for the unique rows
    indices = np.array(np.arange(index+1),dtype=int)
    #flat_data_in_37_array contains: columns 0-31: features. 32: labels. 33:count of feature+label. 34: features as single string (for sorting). 35: index for grouping unique feature vectors 36: zero for unset, 1 for train, 2 for test :
    flat_data_in_37_array =  np.concatenate((flat_data_in_36_array,np.zeros((flat_data_in_36_array.shape[0],1),dtype=int)),axis=1)
    #for i in range(flat_data_in_37_array.shape[0]): print ("check 3 flat_data_in_37_array[",i,",0] = ", flat_data_in_37_array[i,0].astype(int))
    # randomly shuffle the indices
    np.random.shuffle(indices)
    # split the indices into train and test sets
    split_idx = int(len(indices) * 2/3)
    train_idx = np.array(indices[:split_idx],dtype=int)
    train_idx.sort() # will be useful to have these sorted, and makes no difference ML wise.
    test_idx = np.array(indices[split_idx:],dtype=int)
    test_idx.sort()
    print("test_idx.shape = ", test_idx.shape)
    train_data = flat_data_in_37_array # just want to get the right dimensions and data types based on template
    train_data=np.delete(train_data, np.s_[0:train_data.shape[0]],axis=0)   # empty the array
    # use the indices to extract the corresponding rows from the unique rows array
    test_data = train_data             # just want to get the right dimensions and data types based on template
    print("Now test_data should have shape (0,37)  : ", test_data.shape)   
    print("before splitting, train_data.shape, test_data.shape are : ", train_data.shape, test_data.shape)
    numTestRows = int(0)
    i = int(0)
    for test_index in test_idx :
        #print("working on test_index = ",test_index)
        #for i in range(flat_data_in_37_array.shape[0]) :
        if (i < (flat_data_in_37_array.shape[0])) :
            while (int(flat_data_in_37_array[i,35]) <=  int(test_index)) :
                #print ("working on flat_data_in_37_array[",i,"  ] of  ", flat_data_in_37_array.shape[0])
                #print ("about to check whether ",int(train_data[i,35])," > ",  test_index)
                if (int(flat_data_in_37_array[i,35]) ==  int(test_index)) : # if the top row of flat_data_in_37_array should be moved to the test set:
                    numTestRows = numTestRows+1
                    flat_data_in_37_array[i,36] = 2 
                else : # otherwise, send  top row of flat_data_in_37_array to train set:
                    #print("got flat_data_in_37_array[i,35] = ",flat_data_in_37_array[i,35]," which does not match  test_index = ",  test_index)
                    flat_data_in_37_array[i,36] = 1 
                # either way, we're done with the top row:
                i = i + 1
                if (int(i) >= int(flat_data_in_37_array.shape[0])) :
                    break # exit while loop
    print("Count flat_data_in_37_array[i,36] == 0 (should be 0) :", np.count_nonzero(flat_data_in_37_array [:,36].astype(int) == 0))
    print("Count flat_data_in_37_array[i,36] == 1 (train) :",np.count_nonzero(flat_data_in_37_array [:,36].astype(int) == 1) )
    print("Count flat_data_in_37_array[i,36] == 2 (test) :",np.count_nonzero(flat_data_in_37_array [:,36].astype(int) == 2))

    if (i < (flat_data_in_37_array.shape[0]-1)) :
        print ("not done with flat_data_in_37_array! got to i = ",i," out of flat_data_in_37_array.shape[0] = ",flat_data_in_37_array.shape[0])
        flat_data_in_37_array[i:,36]=int(1)
    print("Count flat_data_in_37_array[i,36] == 0 (should be 0) :", np.count_nonzero(flat_data_in_37_array [:,36].astype(int) == 0))
    print("Count flat_data_in_37_array[i,36] == 1 (train) :",np.count_nonzero(flat_data_in_37_array [:,36].astype(int) == 1) )
    print("Count flat_data_in_37_array[i,36] == 2 (test) :",np.count_nonzero(flat_data_in_37_array [:,36].astype(int) == 2))
    # sort by column 36, so we should have some number of training rows followed by numTestRows test rows:
    flat_data_in_37_array=flat_data_in_37_array[np.argsort(flat_data_in_37_array[:,36])]
    train_data = flat_data_in_37_array[:(flat_data_in_37_array.shape[0]-numTestRows)] 
    test_data = flat_data_in_37_array[(flat_data_in_37_array.shape[0]-numTestRows):] 
    #train_data = np.concatenate((train_data, flat_data_in_37_array), axis = 0)    
      
    print("after  splitting, train_data.shape, test_data.shape are : ", train_data.shape, test_data.shape)
    #flat_data_in_37_array contains: columns 0-31: features. 32: labels. 33:count of feature+label. 34: features as single string (for sorting). 35: index for grouping unique feature vectors 36: zero for unset, 1 for train, 2 for test :
    if (str("hello") == str("hello")):
        print ("all is well")
    else: sys.exit("all is not well")
    trainTestIntersect, indicesTrain, indicesTest = np.intersect1d(train_data[:,34].astype(str),  test_data[:,34].astype(str), assume_unique=False,return_indices=True )
    print("shape of trainTestIntersect.shape = ", trainTestIntersect.shape  )
    print(" trainTestIntersect = ", trainTestIntersect)
    print(" indicesTrain       = ", indicesTrain      )
    print(" indicesTest       = ", indicesTest      )
    if (trainTestIntersect.shape[0]):
        sys.exit("all is not well. ,trainTestIntersect.shape[0] should be 0")
    else: print   ("all is well")
    #print(" train_data[indicesTrain,34] = ",  train_data[indicesTrain,34])
    #print(" test_data[indicesTest,34] = ",  test_data[indicesTest,34])
    #if (trainTestIntersect.shape[0]) : 
        #sys.exit("One or more sequences appear in test and train datasets")
    #for i in range(train_data.shape[0]):
    #    print ("testing i = ",i," of ",train_data.shape[0] )
    #    for j in range(test_data.shape[0]):
    #        if (train_data[i,34].astype(str) == test_data[j,34].astype(str) ):
    #            sys.exit("Fatal error. the training feature ",train_data[i,34]," appears also in the test set: ",  test_data[j,34])
            
    
    #for i in range(train_data.shape[0]): print ("check 4 train_data[",i,",0] = ", train_data[i,0].astype(int))
    #for i in range(test_data.shape[0]): print ("check 4 test_data[",i,",0] = ", test_data[i,0].astype(int))
    #test_data_duplicates = test_data; # make sure we get the right format
    #test_data_duplicates=np.delete(test_data_duplicates, np.s_[0:test_data_duplicates.shape[0]],axis=0)   # empty the array
    #train_data_duplicates =  test_data_duplicates # initialize another empty array
    #temp_repeats = test_data_duplicates # initialize another empty array

    #print(" train_data_duplicates.shape, test_data_duplicates.shape are : ", train_data_duplicates.shape, test_data_duplicates.shape)
    train_data = expandRepeats(train_data,33)
    test_data =  expandRepeats( test_data,33)
     
    #train_data_reshaped  = train_data.reshape(train_data.shape[0], 8,4)
    #test_data_reshaped   = test_data.reshape (test_data.shape[0] , 8,4)
    x_train = train_data[:,0:32].astype(int) 
    print(" x_train.shape = ",   x_train.shape)
    x_train = x_train.reshape (x_train.shape[0] , 8,4)
    print(" x_train.shape = ",   x_train.shape)
    y_train = train_data[:,32  ].astype(int) 
    x_test  = test_data [:,0:32].astype(int) 
    x_test = x_test.reshape (x_test.shape[0] , 8,4)
    y_test  = test_data [:,32  ].astype(int) 
    return  x_train, y_train, x_test, y_test

# Load dataset.
# Please change this to the correspond directory. 
# Datasets are saved in the shared Google drive: deep-learning-RNA-structure-prediction/Chih-Fan/TetraloopsDatasets
# In case you are not the owner of this folder, you have to make a copy (either the entire folder or files) to your own drive. 
def prepareDatasetOldWay():
    dict_train = np.load('.//TetraloopsDatasets/noAnno_train_array.npz')  # This is a dictionary, 1513639 entries
    dict_test = np.load('.//TetraloopsDatasets/noAnno_test_array.npz')
    x_train = np.stack(dict_train['arr_0'], axis=0)
    x_train =np.flip( x_train,axis=1) # For some reason the sequences were all in reverse order. Flip to make it more cognitively reasonable.
    y_flt_train = np.load('.//TetraloopsDatasets/noAnno_train_labels.npy')
    y_train = y_flt_train.astype(int)
   
    x_test = np.stack(dict_test['arr_0'], axis=0)
    x_test  =np.flip( x_test ,axis=1)
    y_flt_test = np.load('.//TetraloopsDatasets/noAnno_test_labels.npy')
    y_test = y_flt_test.astype(int)
    return x_train, y_train, x_test, y_test

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



