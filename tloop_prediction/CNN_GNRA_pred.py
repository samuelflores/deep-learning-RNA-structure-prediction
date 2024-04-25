"""""
Kernel size: 8. 1D Convolutional layer
"""""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import math
from numpy import asarray
from numpy import save
from numpy import load
from keras.models import Sequential
from keras.layers import Dense,Conv1D, Conv2D, MaxPooling2D, Dropout, Flatten, Input, MaxPooling1D
from sklearn.preprocessing import StandardScaler
import os
import tempfile
import keras
from keras.optimizers import RMSprop
import matplotlib as mpl

#===========================================================================================
#===========================================================================================

"""""
In machine learning, an epoch refers to one complete pass through the entire training dataset = "times you read a book"
This constant likely represents the number of times the algorithm will iterate through the entire dataset during the training process.

Batch size determines the number of samples that will be used in one iteration (forward and backward pass) of the training process;
Smaller batch sizes consume less memory but might take longer to converge, while larger batch sizes can speed up the training but might require more memory.
假如说，batch size = 5，data = 100 -->  20 iteration --> 1 epoch
"""
EPOCHS = 20
BATCH_SIZE = 16
mpl.rcParams['figure.figsize'] = (15, 10) #rcParams是 Matplotlib 中用来管理默认配置的参数字典。It sets the default size of the figures plotted using Matplotlib to (15, 10) inches.
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] #It assigns the list of default colors to the variable colors. This can be helpful when plotting multiple lines or curves in a graph, ensuring each line gets a distinct color from the cycle.

# Load training dataset.
dict_train = np.load('noAnno_train_array.npz')
'''''
这行代码从名为 'noAnno_train_array.npz' 的 NumPy 压缩文件加载数据。这个文件里可能包含了多个数组。加载后，它被赋值给了名为 dict_train 的变量
np.load()会将noAnno_train_array.npz文件转变成字典储存。用以下可以查看：
print(dict_train.files) #看字典中的键
for key, value in dict_train.items(): #看字典中的每一个键和对应的值；结果来看，这个字典只有一个键，一个值，这个值是个数组。
    print(f"Key: {key}, Value: {value}") #{}内不受' '定义的字符串影响，让电脑知道{}内部是参数，表达式或者函数，因此不会被print成value或者key，而是它们所代表的值
'''''
x_train = np.stack(dict_train['arr_0'], axis=0)
""""
从 dict_train 字典中选择键为 'arr_0' 的数组。这个数组被作为训练数据 x_train，即所有的one hot encoding
np.stack() 函数：它的作用是将一个序列（例如列表、元组、数组等）中的数组沿着指定的轴进行堆叠，生成一个新的数组。
这里使用了 axis=0，表示沿着垂直方向（第一个轴，通常代表样本个数）进行堆叠。也就是说，所有的数组都会沿着垂直方向对齐（想象所有数据像纸张一样摞在一起），形成一个三维的数组。
当需要在深度方向上连接多个数组或张量时，比如处理神经网络中的卷积层输出，或者是处理深度学习中的图像数据集时，np.stack() 能够在新的深度（或通道）轴上堆叠数据。
卷积层（Convolutional Layer）是深度学习神经网络中的一种核心组件，用于处理图像数据或具有类似结构的数据。卷积层通过卷积操作对输入数据进行特征提取。
卷积层的输出是深度学习模型中一个重要的中间结果，通过堆叠多个卷积层，神经网络可以逐渐学习和提取数据中的更加复杂的特征，这有助于模型更好地理解输入数据并完成特定的任务。
"""
y_flt_train = np.load('noAnno_train_labels.npy')
'''''
.npy 文件： 这种文件格式是用于存储单个 NumPy 数组的二进制文件。
当你使用 np.save() 函数保存一个 NumPy 数组时，它会生成一个 .npy 文件。这种文件只能存储单个数组。
通过 np.load('my_array.npy') 加载这个文件，你会得到一个包含原始数组的 NumPy 数组。
.npz 文件： 这种文件格式用于存储多个 NumPy 数组的压缩文件。
当你需要保存多个数组时，可以使用 np.savez() 或 np.savez_compressed() 函数创建一个 .npz 文件。
这个文件包含了多个数组，并且每个数组都可以用一个指定的关键字来表示。通过 np.load('my_arrays.npz') 加载这个文件，你会得到一个类似字典的对象，可以通过关键字访问其中的数组。
'''''
y_train = y_flt_train.astype(int) #.astype(int) 是 NumPy 数组的方法，用于将数组的数据类型转换为整数类型。但是因为label里本身就是整数，所以不会变

# Load training dataset.
dict_test = np.load('noAnno_test_array.npz')
x_test = np.stack(dict_test['arr_0'], axis=0)
y_flt_test = np.load('noAnno_test_labels.npy')
y_test = y_flt_test.astype(int)

#view the results
print('Training features shape:', x_train.shape)
""""
.shape 是用于获取 NumPy 数组 x_train 的形状（shape）信息的方法。它返回一个元组，表示数组在每个维度上的大小。
比如 (m, n, ...)，其中 m、n 等是每个维度上的大小。
如果 x_train 是一个二维数组，它的形状为 (100, 50)，表示它有 100 行和 50 列的数据点
"""""
print('Test features shape:', x_test.shape)
print('Training labels shape:', y_train.shape)
print('Test labels shape:', y_test.shape)

#===========================================================================================
#===========================================================================================
"""""
keras.metrics.XXXX() is a function in Keras used for defining various evaluation metrics when compiling a neural network model in Keras.
"""""

METRICS = [
      keras.metrics.TruePositives(name='tp'), #Counts the number of true positive predictions.
      keras.metrics.FalsePositives(name='fp'), #Counts the number of false positive predictions.
      keras.metrics.TrueNegatives(name='tn'), #Counts the number of true negative predictions.
      keras.metrics.FalseNegatives(name='fn'), #Counts the number of false negative predictions.
'''''
      above give insights into the model's ability to correctly classify instances of each class.
'''''
      keras.metrics.BinaryAccuracy(name='accuracy'), #Computes the accuracy of the model's predictions - overall measure of correct predictions.
      keras.metrics.Precision(name='precision'), #Measures the precision of the model's positive class predictions - precision = TP/(TP+FP) 分母是所有的positive results in prediction
      keras.metrics.Recall(name='recall'), #or sensitivity, Measures the recall of the model's positive class predictions - recall = TP/(TP+FN) 分母是所有的positive results in reality
      keras.metrics.AUC(name='auc'), #Computes the Area Under the Curve (AUC) for ROC or Precision-Recall curves.
'''''
      ROC curves plot the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings for a binary classifier. 
      The AUC-ROC summarizes the classifier's performance across various thresholds for distinguishing between classes.
      In binary classification, models often produce a probability score (ranging between 0 and 1) that indicates the likelihood of an instance belonging to a particular class (usually the positive class).
      To make a concrete prediction, these continuous scores need to be converted into discrete class labels (such as 0 or 1).
      A threshold is applied to these scores; instances with a score above the threshold are predicted as positive, while those below the threshold are predicted as negative.
      Adjusting the threshold affects the trade-off between true positive rate (TPR) and false positive rate (FPR) and, subsequently, the precision and recall of the model.
      Each point on the ROC curve corresponds to a specific threshold used to classify the instances. However, the actual threshold values themselves are not plotted on the ROC curve.
      after determining the optimal threshold, you can apply the selected threshold to the model and the threshold can  evaluate the model's performance.
'''''
      keras.metrics.AUC(name='prc', curve='PR'), # Specifically computes the AUC for the Precision-Recall curve.
      '''''
       Specifically computes the AUC for the Precision-Recall (PR) curve. 
       PR curves plot the precision against the recall at various threshold settings for a binary classifier. 
       Precision measures the accuracy of positive predictions, while recall measures the ability to find all positive instances. 
       The AUC-PR summarizes the model's performance across various thresholds for differentiating positive and negative instances.
       Choosing between AUC-ROC and AUC-PR depends on the specific context of the problem. 
       AUC-ROC is commonly used when evaluating balanced datasets, while AUC-PR is preferred in scenarios with imbalanced datasets or when the positive class is more critical and needs to be accurately identified.
      '''''
]
"""""
这些指标可用于监控模型在训练过程中的性能表现，并根据模型在验证集或测试集上的表现进行评估和比较。
在训练过程中，这些指标将被记录和显示，以便在模型训练过程中监控模型性能的变化。
"""""

#===========================================================================================
#===========================================================================================
neg, pos = np.bincount(y_train)+np.bincount(y_test)
total = neg + pos
"""""

np.bincount() 是 NumPy 中的一个函数，它用于统计非负整数数组中每个值的出现次数。
该函数返回一个数组，数组的索引表示整数值，而数组的值表示对应整数值在输入数组中出现的次数。
np.bincount(y_train)和np.bincount(y_test)分别返回两个数组，比如[3396 4572]和[1701 2283]。它们相加后再分别把第一个和第二个数字赋值给neg和pos
"""""
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

print('Training labels shape:', y_train.shape)
print('Test labels shape:', y_test.shape)

print('Training features shape:', x_train.shape)
print('Test features shape:', x_test.shape)

#===========================================================================================
#===========================================================================================
#下面的函数 make_model1 使用 Keras 的 Sequential API 创建了一个用于二元分类问题的神经网络模型。

def make_model1(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    '''
    函数参数设置：make_model1(metrics=METRICS, output_bias=None)，这个函数接受两个参数：metrics（默认为 METRICS，可能是一个包含评估指标的列表）和 output_bias（默认为 None）。
    output_bias 参数用于初始化模型最后一个密集层的偏差项。
    tf.keras.initializers.Constant(output_bias) 是一个常数初始化器，它会将偏差值设置为 output_bias 这个常数值，即0。
    tf.keras.initializers.Constant()会被用作最后一个密集层的偏差项的常量初始化器。这对于处理类别不平衡问题很有帮助
    '''
    model = Sequential([
    '''
    Keras 的 Sequential 是一种简单的模型构建方式，它允许按顺序逐层添加神经网络层，层之间的数据流是单向的，从输入到输出。
    在机器学习框架中，API 是指定了如何构建、训练和使用模型的一组函数、类和方法。
    Sequential API 允许你一层接一层地添加神经网络层，形成线性堆叠的模型结构。
    除了 Sequential API，Keras 还提供了另一种模型构建方式，即 Functional API。Functional API 允许你创建更为灵活和复杂的模型结构，包括多输入多输出的模型、具有共享层的模型等，它更适合构建复杂的神经网络结构。Functional API 使用更加灵活的连接方式，可以创建非线性拓扑结构的模型。
    Keras 还提供了 Subclassing API，它是基于 Python 类的方式构建模型，这种方式能够提供最大的灵活性，但也相对更加复杂，需要更多的编码工作。
    '''
        # The input shape is 8x4
        # This is the first convolution
        Conv1D(16, 3, strides=1, activation='relu', padding='same',
               input_shape=(8, 4),
               kernel_initializer='he_normal',
               bias_initializer='zeros'),
        '''
        在卷积神经网络中，卷积核是一个小的矩阵，它通过与输入数据进行逐元素相乘并求和的操作来提取特征。卷积核中的每个元素（0或1）都代表着对应位置的权重值，这些权重值用于对输入数据进行加权求和。
        在实际的神经网络中，初始的卷积核通常是随机初始化的。这是因为在训练神经网络的过程中，网络的权重需要不断地进行调整和优化，以使得网络能够学习到输入数据中的特征并逐步提高性能。
        参数:
        Filters（滤波器数量）= 16：指定了该卷积层使用的滤波器数量。每个滤波器可以理解为一个特征检测器，用于提取输入数据中特定类型的特征。16个滤波器表示该层将学习16种不同的特征。
           - 数量通常是根据数据和任务的复杂性来决定的，同时也可以作为模型设计的超参数来调整。
        Kernel Size（卷积核大小）= 3：这是卷积核（滤波器）的窗口大小。对于 1D 卷积，这意味着每个滤波器的窗口长度为3。卷积核与输入数据进行逐元素相乘并求和的过程会在窗口内进行滑动，从而提取局部特征。
           - 当滤波器数量为16、卷积核大小为3时，可以通过一个图像处理的类比来解释：
           - 想象你有一台装置，可以在一张纸上移动一个划分为16个小格的窗口（filter数量）。每个小格子的尺寸是3x3（卷积核大小）。每个小格子都可以捕捉纸上局部区域的不同特征。
           - 现在你开始在一张图片上移动这个窗口（卷积核）进行观察。当你在图片上滑动时，这个窗口（卷积核）会不断改变位置，并通过每个滤镜（滤波器）观察窗口中的内容。每个滤镜会识别和捕捉到不同的图像特征，例如边缘、颜色、纹理等。
           - 16个滤镜（滤波器）代表了对图像各种特征的不同敏感性。随着窗口在图像上的移动，每个滤镜都会探测到不同的局部特征，这些特征的组合最终形成了网络中更高级别的特征表示。整个过程类似于卷积神经网络中的卷积操作，通过一组卷积核（滤波器）对输入数据进行特征提取和转换。
        Strides（步长）= 1：表示卷积核在对输入进行滑动时的步幅大小。在这种情况下，卷积核每次向右滑动一个元素的距离，即步长为1，这意味着卷积操作会逐步对输入数据进行处理。每个滤波器分布在不一样的地方，都学习不一样的东西，因此不会重复学习。
        Padding（填充）= 'same'：填充的方式，'same' 表示使用一定的填充量以保持输入和输出的尺寸一致。在 1D 卷积中，这意味着在输入序列的两端分别填充了一定数量的零元素，以便保持输出与输入相同的长度。
           - 假设你有一段纸条，上面有一串文字。你想用一个窗口（卷积核）在这段文字上滑动并捕捉文字中的一些模式。
           - 现在你把这个窗口放在纸条上，窗口大小与纸条上的文字一样长。但是，你希望在窗口移动时，文字的每个字符都被完全覆盖，这样你才能够捕捉到所有可能的特征。
           - 如果你不对纸条进行任何填充，那么当窗口移动到纸条的两端时，窗口内文字的一部分将会消失。这样你就无法完全观察窗口中的所有内容。
           - 现在，如果你使用了填充（padding），就相当于在纸条的两端各贴了一些额外的空白纸，使得文字在窗口内时能够完全展现。这样，当你移动窗口时，文字的每个字符都能够完整地呈现在窗口中，你就可以捕捉到更多的信息，不会丢失任何数据。
           - 在卷积神经网络中，'same' 填充方式就是为了确保在卷积操作时输入和输出的尺寸一致。通过在输入序列的两端填充适当数量的零元素，使得卷积操作时能够完全覆盖输入数据的每个部分，不丢失信息，从而保持输出与输入相同的长度。
        'relu' 是一个激活函数，这是一种常用的非线性激活函数，有助于引入网络的非线性特性。
           - 激活函数是神经网络中的一个重要组成部分，它负责在神经元的输出上应用非线性变换。激活函数将神经元的输入信号转换为输出信号，并为神经网络引入非线性特性，从而使其能够学习和表示复杂的模式和函数关系。
              - 非线性激活函数在神经网络中的作用是为了增加模型的表达能力。如果没有非线性激活函数，多个线性变换的组合仍然会得到一个线性变换。在深度神经网络中，层与层之间只进行线性变换将导致整个网络的表达能力大大降低。
              - 线性变换将输入与权重相乘并加上偏差，这会产生一个线性关系。但是，通过应用非线性激活函数，网络可以学习到更加复杂的特征和模式，从而提高网络的表示能力。例如，ReLU（Rectified Linear Unit）是一种常用的非线性激活函数，它可以在输入大于零时保持不变，小于零时变为零。这种非线性特性使得神经网络可以学习到非线性的特征映射，更好地拟合复杂的数据模式。
           - 'relu'的作用是：当输入为正时，输出与输入相同；当输入为负时，输出为零。当输入为0时，ReLU激活函数输出为0。f(x) = max(0,x)，x为输入
        kernel_initializer='he_normal' 是一种权重(weights)初始化方法，用于初始化神经网络层中的卷积核（滤波器）或全连接层的权重。
           - "He_normal" "He_normal" 是一种初始化权重的策略，它的初始化方法会从正态分布中随机地初始化权重，并且权重的标准差会根据网络中前一层的神经元数量进行缩放。这样的做法有助于保持每一层激活值的方差相对稳定，从而更好地在网络中传播信号并加速训练过程。    
           - 在一个神经网络中，每个神经元都与下一层的每个神经元（如果是全连接层）或部分神经元（如果是卷积层或池化层）相连接。每条连接都有一个相关联的权重，这个权重代表了两个相连神经元之间连接的强度或权重值。
           - 当神经网络进行训练时，权重是需要不断更新和调整的，这个过程就是通过反向传播算法来实现的。在训练期间，网络试图调整这些权重，以最小化损失函数，使得网络的预测结果与实际结果尽可能接近。      
        '''
        MaxPooling1D(pool_size=2, strides=2),
        '''''
        池化层的作用在于对数据进行降维和特征提取，而MaxPooling（最大池化）是其中一种常用的池化方式之一。
        MaxPooling1D 是一种池化（Pooling）层，用于一维数据的特征压缩和提取。它可以将输入的一维数据按照指定的池化窗口大小进行区域划分，并从每个区域中提取最大值，从而实现数据的降维和特征提取。
        它之所以选择提取最大值，是因为在很多情况下，保留区域中最大的特征值可以更好地保留主要特征信息（感觉有点类似dimensional reduction)。
        - 最大值通常代表着某个特征在局部区域的最显著表现。通过保留最大值，池化层更有可能捕捉到局部区域最重要的特征，这对于识别和提取重要特征非常有用。
        - 保留最大值降低了计算的复杂度。
        - 最大池化在一定程度上保留了平移不变性。
        池化层还有其他类型，比如平均池化（Average Pooling），它是从每个区域中取平均值。在不同的场景下，不同类型的池化方式可能会更适合于特定的数据和任务。
        在 CNN（卷积神经网络）进行二元分类预测时，MaxPooling1D 层通常被用来在卷积操作后进行特征压缩和提取，以提高模型的泛化能力和减少过拟合的风险。
        参数：
        pool_size=2：表示池化窗口的大小，即在进行池化操作时每次考虑的区域大小。在这里，池化窗口大小为2，意味着每次会取2个相邻的数据进行池化操作。
        strides=2：表示池化窗口的步幅，即池化操作在输入数据上滑动的步长。在这里，步幅为2，意味着每次池化操作后，窗口向右滑动2个数据点的距离。
        比如数据[1,3,5,2,7,4,6,8,9,0]，经过 MaxPooling1D(pool_size=2, strides=2) 操作后，会将输入数据按照窗口大小2进行划分，然后每次从划分区域中取最大值。因此，经过池化操作后的输出是[3,5,7,8,9]
        '''''
        Dropout(0.2),
        '''
        Dropout 是一种正则化技术，旨在减少神经网络的过拟合(over-fitting)。
        在训练过程中，Dropout 会随机地（以一定的概率）将一部分神经元（节点）的输出置为0，即“丢弃”这些神经元，以减少神经元之间的依赖关系。
        在这里，Dropout(0.2)表示应用了 Dropout 技术，丢弃了神经元的比例为 20%。也就是说，对于每个训练样本，在进行神经网络的前向传播过程中，每个神经元都有 20% 的概率被随机地丢弃，其输出值被设为0。
        这种操作有助于降低神经网络对特定神经元的过度依赖，从而提高网络的泛化能力和抗过拟合能力。在测试或推理阶段，Dropout 是不会被应用的，而是用于训练过程中。
        例如，如果我们观察隐藏层中第一个神经元的输出，在一次训练迭代中可能是：
        - 训练迭代1：神经元 1 的输出为 0 - drop out了
        - 训练迭代2：神经元 1 的输出为 3.2
        - 训练迭代3：神经元 1 的输出为 0 - drop out了
        - ...以此类推
        '''
        Conv1D(32, 3, strides=1, activation='relu', padding='same',
               kernel_initializer='he_normal',
               bias_initializer='zeros'),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        Flatten(),
        '''
        这段代码中重复使用了两个 Conv1D 层，尽管它们的过滤器数量不同。这样做的目的可能是为了构建更深层次的特征提取器。
        通常，在卷积神经网络（CNN）中，通过堆叠多个卷积层可以使模型学习到更加复杂和抽象的特征。
        第一个 Conv1D 层通常会捕捉一些底层的特征，而第二个 Conv1D 层可以利用第一个层的输出来进一步提取更高级别的特征。
        在网络的后续部分，这些不同层次、不同级别的特征会合并在一起，并由后续的全连接层或其他层进一步学习和整合这些特征。例如通过更深层的网络结构将这些特征结合起来，使得网络能够更好地学习数据中的复杂模式和特征。
        Flatten()：用于将多维数据展平为一维的层（比如变成一个很长的list)。
        - 在卷积神经网络中，卷积层和池化层之后一般会连接一个或多个全连接层（Dense 层），而这些全连接层的输入需要是一维的。
        - 因此，在进入全连接层之前，需要将卷积层或池化层输出的多维数据展平为一维数据，这时候就会使用 Flatten() 层。展平后的数据会成为全连接层的输入。
        '''
        #  neuron hidden layer
        Dense(128, activation='relu', bias_initializer=output_bias),  # 8/2(maxpooling)=4, 32*4 = 128
        Dropout(0.2),
        '''
        密集层（Dense Layer）是神经网络中常用的一种层类型，它也被称为全连接层。
        该层中的每个神经元与前一层的所有节点（或神经元）连接，每个连接都有一个权重，这样每个神经元的输出就是前一层神经元的加权和，再加上一个偏差（bias）。这个层会对输入数据进行线性变换和非线性变换。
         - 在这段代码中，第一个 Dense 层有 128 个神经元。而在卷积层 Conv1D(32, 3)，参数中的 32 表示该卷积层使用了 32 个不同的卷积核（即过滤器数量），每个过滤器在卷积过程中负责检测输入数据中的不同特征。因此，在卷积层中，并不直接使用神经元这个术语来表示网络的结构。神经元的概念更常见于全连接（Dense）层中。
         - 使用 'relu' 激活函数。这意味着该层会将输入数据进行线性变换，并将其传递给 128 个神经元，每个神经元都将对输入数据进行加权和处理，并应用 'relu' 非线性激活函数。
        '''
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for class ('not GNRA') and 1 for the other ('GNRA')
        Dense(1, activation='sigmoid')  # Sigmoid for binary question.
        '''
        第二个 Dense 层只有一个神经元，使用 'sigmoid' 激活函数。这个层是一个输出层，因为它只有一个神经元，输出的值范围在 0 到 1 之间。
        通常，对于二元分类问题，这个值可以被看作是模型对于某个类别的预测概率。例如，在这里，0 可能代表 'not GNRA' 类别，而 1 可能代表另一个类别 'GNRA'。
        sigmoid 激活函数将确保输出值在 0 到 1 之间，并且可以被解释为概率值。
        '''
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # optimizer=RMSprop(lr=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)
    ''''
    这段代码是用来编译神经网络模型的步骤，它指定了模型训练时使用的优化器、损失函数和评估指标。
    参数：
    optimizer
    - 优化器决定了模型在训练过程中如何更新权重以最小化损失函数。选择合适的优化器对模型的训练效果和速度有很大的影响。
    - optimizer=keras.optimizers.Adam(learning_rate=1e-3)：这里选择了 Adam 优化器作为模型的优化器。
    - Adam 是一种常用的优化算法，通过自适应学习率调整和动量来有效地优化神经网络的权重。
    - 学习率是深度学习中的一个重要超参数，用于控制模型在每次迭代（每次权重更新）中对损失函数梯度的调整幅度。它决定了模型参数更新的速度和方向。
    - 学习率越大，模型参数更新的步长就越大，模型可能更快地收敛，但也容易跳过最优解或者震荡；学习率越小，模型参数更新的步长较小，收敛速度较慢，但更稳定，有可能更容易收敛到较好的解。
    - learning_rate=1e-3 表示学习率的大小为 0.001。它表示在每次权重更新时，模型会按照损失函数梯度的 0.001 倍来调整参数。
    loss
    - 损失函数是用来衡量模型预测结果与实际标签之间的差异。
    - 例如，假设你有一篮水果，你对其中的每个水果都预测它是苹果或橙子。然后，你核对每个水果的真实标签，如果你的预测和实际水果不符，你就会记录下这个错误。这些错误的数量和程度就是损失函数的值。
    - loss=keras.losses.BinaryCrossentropy()：常见的损失函数包括交叉熵损失函数（Cross-Entropy Loss）。在二元分类问题中，二元交叉熵可用于衡量模型输出概率分布与真实标签的相似度。在多类分类问题中，分类交叉熵则用于衡量多分类模型输出与真实标签之间的差异。
    metrics
    - metrics=metrics：这里使用了一个列表 metrics，包含了在模型训练过程中要评估的指标，如准确率、精确度、召回率、AUC 等。
    - 这些指标用于衡量模型在训练过程中的性能表现。
    '''''

    return model

#===========================================================================================
#===========================================================================================
#设计第二种模型，最后比较两个模型的好坏
def make_model2(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = Sequential([
        # The input shape is 8x4
        # This is the first convolution
        Conv1D(64, 3, strides=1, activation='relu', padding='same',
               input_shape=(8, 4),
               kernel_initializer='he_normal',
               bias_initializer='zeros'),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        Conv1D(32, 3, strides=1, activation='relu', padding='same',
               kernel_initializer='he_normal',
               bias_initializer='zeros'),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        Flatten(),
        #  neuron hidden layer
        Dense(128, activation='relu', bias_initializer=output_bias),  # 8/2(maxpooling)=4, 32*4 = 128
        Dropout(0.2),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for class ('not GNRA') and 1 for the other ('GNRA')
        Dense(1, activation='sigmoid')  # Sigmoid for binary question.
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # optimizer=RMSprop(lr=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

#===========================================================================================
#===========================================================================================

def plot_loss(history, label, n):
  # Use a log scale on y-axis to show the wide range of values.
  plt.semilogy(history.epoch, history.history['loss'],
               color=colors[n], label='Train ' + label)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

'''
plt.semilogy() 创建了一个以对数刻度显示y轴的图表。
history 在机器学习中通常用于跟踪模型训练过程中的指标变化。它可能包含了一系列的指标，其中 epoch 是一个整数列表，记录了训练时的每个 epoch 的索引，而 history 字典中的 'loss' 键对应着每个 epoch 上的损失值。除了loss外，还有别的键值对，如prc, predicions, recall等。
- history.epoch作为x轴，表示训练的时期（epoch）
- history.history['loss']，表示每个时期训练过程中损失函数的值。
color=colors[n] 表示选择一个特定颜色来绘制这条线
而label='Train ' + label则是为这条曲线设置标签，表示训练集上的损失函数变化
'''

#===========================================================================================
#===========================================================================================

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix')
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Non-GNRA Detected (True Negatives): ', cm[0][0])
  print('Non-GNRA Incorrectly Detected (False Positives): ', cm[0][1])
  print('GNRA Missed (False Negatives): ', cm[1][0])
  print('GNRA Detected (True Positives): ', cm[1][1])
  print('Total GNRA: ', np.sum(cm[1]))

'''
labels代表真实的类别标签，predictions代表模型预测的标签，p是阈值。它计算混淆矩阵并将其可视化。
confusion_matrix(labels, predictions > p)：通过传入真实标签和根据预测概率和阈值生成的二元预测值，计算混淆矩阵。
- 只有当预测概率大于阈值 p 的情况下，对应的预测值才会被视为正类别;反之为负类别。
- cm的输出值会是混淆矩阵的值。混淆矩阵是一个二维数组，其中行表示实际类别，列表示预测类别，如[[1680   21]
                                                                             [ 129 2154]]
0 和 1 分别表示矩阵中的行和列索引，因此cm[0][0]锁定的是第0行，第一列的数字，也就是实际标签为0（负类别）但是预测结果为1（正类别）的数量
'''

#===========================================================================================
#===========================================================================================

def plot_metrics(history1, history2):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history1.epoch, history1.history[metric], color=colors[0], label='Conf1_Train')
    plt.plot(history1.epoch, history1.history['val_' + metric],
             color=colors[0], linestyle="--", label='Conf1_Test')
    plt.plot(history2.epoch, history2.history[metric], color=colors[1], label='Conf2_Train')
    plt.plot(history2.epoch, history2.history['val_' + metric],
             color=colors[1], linestyle="--", label='Conf2_Test')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend();

'''
这段代码中的 for n, metric in enumerate(metrics) 是一个循环语句，它用于迭代一个包含多个指标的列表 metrics。
- enumerate() 函数用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列，同时列出数据和数据下标。
- 在这里，enumerate(metrics) 将列表 metrics 中的元素以及它们的索引位置配对，每次循环迭代时，n 表示当前元素的索引，metric 表示当前元素的值。
plt.subplot(2,2,n+1) 中的 n+1 控制子图的位置。plt.subplot() 用于在指定的网格中创建子图，参数 (2, 2, n+1) 指定了一个2x2的网格，n+1 是子图在该网格中的位置编号。
- 在循环中，n 的值从0开始，但子图编号是从1开始的，因此 n+1 的作用是将从0开始的索引转换为从1开始的子图编号。这样可以将不同的子图放置在指定网格的不同位置上。
在 Keras 中，'val_' + metric 表示模型在验证集上相应指标的值。
- 通常，在训练过程中，除了在训练集上计算指标外，还会在验证集上计算相同或类似的指标。因此，'val_' + metric 中的 'val_' 前缀表示这是在验证集上的指标。这种约定用于区分训练集和验证集的指标。
plot.ylim()的作用是设置loss figure纵坐标（y轴）的范围。它返回一个包含两个值的列表或元组，分别表示 y 轴范围的下限和上限。
- 具体来说，它将纵坐标的下限设为 0
- plt.ylim()[1] 中的 [1] 表示取这个列表或元组中的第二个值，即当前 y 轴范围的上限值。
- 实际上，这个上限值是在图表显示时由 Matplotlib 自动计算的。当没有指定 y 轴范围时，Matplotlib 会根据数据的范围自动确定合适的坐标轴范围，确保数据都能够完整地显示在图表中。
'''

#===========================================================================================
#===========================================================================================

def plot_roc(name, labels, predictions, **kwargs):
  fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)  # fpr=false positive rate = fp/(fp+tn), tpr=true positive rate = tp/(tp+fn)

  plt.rcParams['font.size'] = '16'
  plt.plot(100*fpr, 100*tpr, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives rate = fp/(fp+tn)')
  plt.ylabel('True positives rate = tp/(tp+fn)')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

'''''
roc_curve 函数会利用给定的概率值（predictions）和真实标签（labels）计算不同阈值下的 FPR 和 TPR。这允许我们绘制 ROC 曲线，以展示不同阈值下的模型性能。
sklearn.metrics.roc_curve 函数的返回值顺序是固定的，首先是 FPR（假阳率），然后是 TPR（真阳率），最后是阈值（不使用阈值（用_表示））。
ax = plt.gca()和ax.set_aspect('equal')用于获取当前图形的轴对象（ax）并设置其纵横比相等（即使两个轴的比例相同）。这样做有助于确保在绘制 ROC 曲线时 x 轴和 y 轴的比例正确显示，使得图像更易于理解。
**kwargs 是 Python 中用来表示关键字参数的一种语法。这个语法允许你在函数中传递可变数量的关键字参数。
- kwargs 实际上是一个字典。
- 在上面的例子中，**kwargs 可以用来传递 plot_roc 函数中除了 name, labels, predictions 之外的其他参数，这些参数会以字典的形式保存在 kwargs 中。
- 比如，在后续的代码中，用了color=colors[1], linestyle='--'。这些就会自动被plt.plot()考虑到。
'''''

#===========================================================================================
#===========================================================================================

def plot_prc(name, labels, predictions, **kwargs):  #precision = tp / (tp + fp), recall = tp / (tp + fn)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.rcParams['font.size'] = '9'
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall = TP / (TP + FP)')
    plt.ylabel('Precision = TP / (TP + FN)')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

#===========================================================================================
#===========================================================================================

# retrain with class weights
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

''''
这段代码中计算了用于样本类别加权的权重。这里的权重是指损失函数中的样本权重，它们用于调整模型训练过程中的损失计算，而不是直接应用于神经元权重。神经元权重通常是在网络学习过程中通过优化算法（例如梯度下降）进行调整的。类别权重则是在损失函数中对不同类别样本的损失进行加权处理。
在类别不平衡的情况下，损失函数中会使用这些类别权重来赋予少数类别更高的重要性。
weight_for_0 和 weight_for_1 是两个权重，用于调整类别 0 和类别 1 的样本在训练中的重要性。weight_for_0 和 weight_for_1 分别表示类别 0 和类别 1 的权重。
这里的计算方式是基于每个类别的样本数量，通过加权使得每个类别对模型训练的影响更加平衡，以解决数据集类别不平衡的问题。
为什么要加权：
- 想象一下医学诊断中的情况：假设你有一个罕见的疾病，发病率是 1%，你的模型要预测这种疾病。你有一个数据集，里面有 99% 的健康样本，1% 的患病样本。如果你的模型只是简单地预测每个人都是健康的，那么它的准确率就会达到 99%。
- 比如有10个负样本和2个正样本的情况。一个模型预测了两个样本：一个是负样本，一个是正样本，它们的预测结果分别为[0.8, 0.2]（正样本的预测概率较低）。假设正样本的权重是5，负样本的权重是1。如果没有加权，损失计算为负样本：0.8（误差较小）；正样本：0.2（误差较大）；整体损失：(0.8 + 0.2) / 2 = 0.5。但是如果加权，负样本损失为0.8 * 1 = 0.8，正样本损失为0.2 * 5 = 1，整体损失：(0.8 + 1) / 2 = 0.9。这种加权可以让模型更加关注少数类别的样本。
如何计算权重：
- 权重的计算通常涉及样本的相对比例。
- 在这个例子中，权重的计算是通过总样本数和各个类别的样本数量来实现的。
- 作者首先将总样本数除以2，这是为了将权重缩放到一个合适的范围内（基准权重）。
- 然后除以正负样本中的样本数量；可以想象为平均一个样本，占基准权重的多少。因此这个由每个样本预测而来的结果，也应该乘上这个权重。
{:.2f} 是 Python 中的字符串格式化方法。在这个例子中，:.2f 的含义是格式化成小数点后两位的浮点数。在这个格式中，. 表示小数点，2 表示保留两位小数，f 表示浮点数。
format() 方法用于将相应的值插入到字符串中，并根据格式进行格式化，{:.2f} 中的 .2f 就是用来指定保留小数点后两位的格式。
'''''

#===========================================================================================
#===========================================================================================

weighted_model1 = make_model1()
#weighted_model.load_weights(initial_weights)

weighted_history1 = weighted_model1.fit(
    x_train,
    y_train,
    validation_data = (x_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,

    # The class weights go here
    class_weight={0: 1.17, 1: 0.87})

'''''
weighted_history1 = weighted_model1.fit() 是使用 Keras 模型对象 weighted_model1 对训练数据进行拟合（训练）的步骤。
"拟合" 是指使用给定的数据训练模型，使其能够尽可能准确地表示或逼近数据集中的模式、特征或关系。在机器学习和深度学习中，拟合指的是根据数据对模型参数进行调整，以便模型能够对数据进行有效预测或分类。
这里就是指要把weighted_model1根据权重优化预测结果，并存储在weighted history1里。
'''''

#===========================================================================================
#===========================================================================================

weighted_model2 = make_model2()
#weighted_model.load_weights(initial_weights)

weighted_history2 = weighted_model2.fit(
    x_train,
    y_train,
    validation_data = (x_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,

    # The class weights go here
    class_weight={0: 1.17, 1: 0.87})

#===========================================================================================
#===========================================================================================
#below are all results
plot_metrics(weighted_history1, weighted_history2)

train_predictions_weighted1 = weighted_model1.predict(x_train, batch_size=BATCH_SIZE)
test_predictions_weighted1 = weighted_model1.predict(x_test, batch_size=BATCH_SIZE)

''''
在上面的代码中，可能会对训练集和测试集都进行预测是因为在训练过程中，通常会在每个 epoch 结束时对训练集和测试集进行评估，以监测模型在训练和测试数据上的表现。
这有助于了解模型是否过拟合（在训练集上表现良好但在测试集上表现不佳）或者欠拟合（在训练和测试集上表现都不佳）。
所以这段代码可能是为了在训练过程中监测模型在训练集和测试集上的表现。
'''''

train_predictions_weighted2 = weighted_model2.predict(x_train, batch_size=BATCH_SIZE)
test_predictions_weighted2 = weighted_model2.predict(x_test, batch_size=BATCH_SIZE)

# Results of test data set.
weighted_results1 = weighted_model1.evaluate(x_test, y_test,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model1.metrics_names, weighted_results1):
  print(name, ': ', value)
print()

plot_cm(y_test, test_predictions_weighted1)

# Results of test data set.
weighted_results2 = weighted_model2.evaluate(x_test, y_test,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model2.metrics_names, weighted_results1):
  print(name, ': ', value)
print()

plot_cm(y_test, test_predictions_weighted2)

plot_roc("Conf1_Train Weighted", y_train, train_predictions_weighted1, color=colors[0])
plot_roc("Conf1_Test Weighted", y_test, test_predictions_weighted1, color=colors[0], linestyle='--')
plot_roc("Conf2_Train Weighted", y_train, train_predictions_weighted2, color=colors[1])
plot_roc("Conf2_Test Weighted", y_test, test_predictions_weighted2, color=colors[1], linestyle='--')


plt.legend(loc='lower right');

plot_prc("Conf1_Train Weighted", y_train, train_predictions_weighted1, color=colors[0])
plot_prc("Conf1_Test Weighted", y_test, test_predictions_weighted1, color=colors[0], linestyle='--')
plot_prc("Conf2_Train Weighted", y_train, train_predictions_weighted2, color=colors[1])
plot_prc("Conf2_Test Weighted", y_test, test_predictions_weighted2, color=colors[1], linestyle='--')
# plt.xlim([0, 1.2])
plt.ylim([0, 1.2])

plt.legend(loc='upper right');