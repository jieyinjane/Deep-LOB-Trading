# load packages
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from datetime import datetime

import gc
from sklearn import preprocessing

# limit gpu memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
       # Currently, memory growth needs to be the same across GPUs
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
       # Memory growth must be set before GPUs have been initialized
       print(e)

# set random seeds
np.random.seed(1)
tf.random.set_seed(2)

# labelling
def wealth_change_true(bid_price, ask_price, spread):
    wct = []
    TC = 1  # transaction cost ratio
    pos_inter = 20  # position interval
    beta1 = 1 - spread*4 
    beta12 = 1 + spread*2 
    beta2 = 1 + spread*4  
    beta21 = 1 - spread*2

    for index in range(int(len(bid_price) - pos_inter)):
        # short direction, first bid then ask price
        if min(ask_price[index: index + pos_inter]) <= beta1 * bid_price[index] and max(
                ask_price[index: index + pos_inter]) >= beta12 * bid_price[index]:
            if np.where(ask_price[index: index + pos_inter] <= beta1 * bid_price[index])[0][0] < \
                    np.where(ask_price[index: index + pos_inter] >= beta12 * bid_price[index])[0][0]:
                c_1t = 2 - beta1 * TC
            else:
                c_1t = 2 - beta12 * TC
        elif min(ask_price[index: index + pos_inter]) <= beta1 * bid_price[index]:
            c_1t = 2 - beta1 * TC
        elif max(ask_price[index: index + pos_inter]) >= beta12 * bid_price[index]:
            c_1t = 2 - beta12 * TC
        else:  
            c_1t = (2 * bid_price[index] - min(ask_price[index: index + pos_inter]) * TC) / bid_price[index]

        # long direction, first ask then bid price
        if max(bid_price[index: index + pos_inter]) >= beta2 * ask_price[index] and min(
                bid_price[index: index + pos_inter]) <= beta21 * ask_price[index]:
            if np.where(bid_price[index: index + pos_inter] >= beta2 * ask_price[index])[0][0] < \
                    np.where(bid_price[index: index + pos_inter] <= beta21 * ask_price[index])[0][0]:
                c_2t = beta2 / TC
            else:
                c_2t = beta21 / TC
        elif max(bid_price[index: index + pos_inter]) >= beta2 * ask_price[index]:
            c_2t = beta2 / TC
        elif min(bid_price[index: index + pos_inter]) <= beta21 * ask_price[index]:
            c_2t = beta21 / TC
        else:
            c_2t = max(bid_price[index: index + pos_inter])/ (ask_price[index] * TC)

        wct.append(np.argmax([c_1t * int(c_1t > 1.001), c_2t * int(c_2t > 1.001), 1]))
    return wct

def prepare_x(data):
    df1 = data[:, :40]
    return np.array(df1)

def get_label(data):
    lob = data[:, -1]
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)
    dataY = dY[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D),dtype='float16')
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX.reshape(dataX.shape + (1,)), dataY

def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[[1],[2],[3],[2]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed

def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))

    node = 16
    # build the convolutional block
    conv_first1 = Conv2D(node, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(node, (5, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    conv_first1 = Conv2D(node, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(node, (5, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(node, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(node, (5, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    node2 = 16
    # build the inception module
    convsecond_1 = Conv2D(node2, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(node2, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(node2, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(node2, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(node2, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    # build the last LSTM layer
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(optimizer=adam, loss=[multi_category_focal_loss1(alpha=[[1],[1],[1]], gamma=2)], metrics=['accuracy'])
    # model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model


def DCNN_training(ratio,i):
    col = ['B1P', 'B1V', 'S1P', 'S1V', 'B2P', 'B2V','S2P', 'S2V', 'B3P', 'B3V', 'S3P', 'S3V', 'B4P', 'B4V', 'S4P', 'S4V', 'B5P', 'B5V', 'S5P', 'S5V',
           'B6P', 'B6V', 'S6P', 'S6V', 'B7P', 'B7V', 'S7P', 'S7V', 'B8P', 'B8V', 'S8P', 'S8V','B9P', 'B9V', 'S9P', 'S9V', 'B10P', 'B10V', 'S10P', 'S10V']
    df = pd.HDFStore('/lustre/project/Stat/s1155133513/simulation/test_long.h5')
    data = df['ob'].reset_index(drop=True)
    df.close()
    dimension = data.shape[0]
    N = 41
    dec_train = data.iloc[int(dimension *(ratio-0.45)):int(dimension * ratio), 1:N].reset_index(drop=True)
    bp = dec_train.iloc[:, 0].reset_index(drop=True)
    ap = dec_train.iloc[:, 2].reset_index(drop=True)
    spread_ratio = max(np.mean(np.array(ap) / np.array(bp)) - 1, 0.001)
    print(spread_ratio)
    train_label = wealth_change_true(bp, ap, spread_ratio)

    dec_valid = data.iloc[int(dimension * ratio):int(dimension * (ratio+0.025)), 1:N].reset_index(drop=True)
    bp = dec_valid.iloc[:, 0].reset_index(drop=True)
    ap = dec_valid.iloc[:, 2].reset_index(drop=True)
    valid_label = wealth_change_true(bp, ap, spread_ratio)

    dec_test = data.iloc[int(dimension * (ratio+0.025)):int(dimension * (ratio+0.05)), 1:N].reset_index(drop=True)
    bp = dec_test.iloc[:, 0].reset_index(drop=True)
    ap = dec_test.iloc[:, 2].reset_index(drop=True)
    test_label = wealth_change_true(bp, ap, spread_ratio)

    pos_inter = 20
    mp_test = dec_test.values[:-pos_inter,:]
    print(dec_train.shape)
    print(sum(np.array(train_label) == 0) / len(train_label))
    print(sum(np.array(train_label) == 1) / len(train_label))

    ss = preprocessing.StandardScaler().fit(dec_train)
    dec_train = pd.DataFrame(ss.transform(dec_train), columns=col).values[:-pos_inter,:]
    dec_valid = pd.DataFrame(ss.transform(dec_valid), columns=col).values[:-pos_inter,:]
    dec_test = pd.DataFrame(ss.transform(dec_test), columns=col).values[:-pos_inter,:]

    # extract limit order book data from the dataset
    train_lob = prepare_x(dec_train)
    valid_lob = prepare_x(dec_valid)
    test_lob = prepare_x(dec_test)
    del dec_train,dec_valid,dec_test,data
    gc.collect()

    # prepare training data. We feed past T observations into our algorithms and choose the prediction horizon.
    T = 50
    trainX_CNN, trainY_CNN = data_classification(train_lob, train_label, T)
    trainY_CNN = np_utils.to_categorical(trainY_CNN, 3) 
    # prepare valid and test data.
    validX_CNN, validY_CNN = data_classification(valid_lob, valid_label, T)
    validY_CNN = np_utils.to_categorical(validY_CNN, 3)

    testX_CNN, testY_CNN = data_classification(test_lob, test_label, T)
    testY_CNN = np_utils.to_categorical(testY_CNN, 3)
    
    del train_lob, train_label, valid_lob, valid_label,test_lob, test_label
    gc.collect()
    
    deeplob = create_deeplob(T, N-1, 32)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='auto')
    checkpoint_filepath = './model_check/weights'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_weights_only=True,
                                                                monitor='val_loss',
                                                                mode='auto',
                                                                save_best_only=True)
    deeplob.fit(trainX_CNN, trainY_CNN, epochs=200, batch_size=128, verbose=2, validation_data=(validX_CNN, validY_CNN),
                callbacks=[model_checkpoint_callback, early_stopping])  #, class_weight=class_weights)
    deeplob.load_weights(checkpoint_filepath)
    deeplob.save('./model_save/my_model.h5')
      
    # evaluate the model
    predictions = deeplob.predict(testX_CNN)
    pd.DataFrame(predictions).to_csv('/lustre/project/Stat/s1155133513/simulation/test_y_lf'+str(2*i+1)+'.csv')
    results = np_utils.to_categorical(np.argmax(predictions, axis=1), 3)
    print(classification_report(testY_CNN, results, target_names=['0', '1', '2']))
    
    print("Evaluate")
    result = deeplob.evaluate(testX_CNN, testY_CNN,verbose=0)
    print(dict(zip(deeplob.metrics_names, result)))
    
    del trainX_CNN, trainY_CNN, validX_CNN, validY_CNN, testX_CNN, testY_CNN
    gc.collect()
    
    return predictions, mp_test, spread_ratio


if __name__ == '__main__':
    for i in range(21):
        ratio = 0.025*i+0.45
        predictions, mp_test, spread_ratio = DCNN_training(ratio,i)
