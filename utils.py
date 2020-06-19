 # General
import h5py
import numpy as np
import random
import os
from glob import glob
from tqdm import tqdm

# Modeling and training
from keras import optimizers, losses, activations, models

from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, SpatialDropout1D, TimeDistributed, Bidirectional, LSTM

from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

# Data Analisis
import matplotlib.pyplot as plt
import datetime as dt
import collections
from scipy.fftpack import fft, fftfreq


WINDOW_SIZE = 100
BATCH_SIZE = 10
def rescale_array(X):
    X = X / 20
    X = np.clip(X, -5, 5)
    return X


def aug_X(X):
    scale = 1 + np.random.uniform(-0.1, 0.1)
    offset = np.random.uniform(-0.1, 0.1)
    noise = np.random.normal(scale=0.05, size=X.shape)
    X = scale * X + offset + noise
    return X

def extract(arr, muestras):
    middle = int(muestras / 2);
    p_m = int(arr.shape[0] / 2.0);
    inx_first = p_m - middle;
    res = arr[inx_first:muestras + inx_first]
    return res
            
def gen(dict_files, aug, LENGTH_BASE):
    while True:
        record_name = random.choice(list(dict_files.keys()))
        batch_data = dict_files[record_name]
        all_samples = batch_data['x']
        for i in range(BATCH_SIZE):
            start_index = random.choice(range(all_samples.shape[0]-WINDOW_SIZE))

            X = all_samples[start_index:start_index+WINDOW_SIZE, ...]
            Y = batch_data['y'][start_index:start_index+WINDOW_SIZE]
            #print_s('X antes --->', X)
            #X = np.resize(X,(X.shape[0], LENGTH_BASE, 1))
            X = np.array([extract(s, LENGTH_BASE) for s in X])
            #print_s('X despues --->', X)
            Y = np.array([stages2(s) for s in Y])
            X = np.expand_dims(X, 0)
           
            Y = np.expand_dims(Y, -1)
            Y = np.expand_dims(Y, 0)
            X = rescale_array(X)
            yield X,Y
            


def print_s(title, data):
    print('_______________________________________________')
    print(data.shape)
    print(title + ':')
    print(data)
    
    
def stages2(stage):
    stages = [1,2,3,4,5]
    if stage in stages:
        return 1
    else:
        return 0

# def normalization(serie):
#     scaler = MinMaxScaler(feature_range=(-1, 1)) # between -1,1 because serie has negative values 
#     scaler.fit(serie)
#     normalized = scaler.transform(serie)
#     return normalized
      
def chunker(seq, size=WINDOW_SIZE):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
  

def plot_file(path):
    data = np.load(path+".npz")

    x = data['x']
    y = data['y']
  
    print('shape X')
    print('x', x.shape)
    print('shape Y')
    print('y', y.shape)

    ramdom_sample =random.randrange(y.shape[0])    
    print('random_sample')
    print(ramdom_sample)
  
    list_x = x[ramdom_sample, ...]
    print_s("list_x",list_x)
    list_y = y[ramdom_sample]
    print_s("list_y",list_y)
    fig_1 = plt.figure(figsize=(17, 6))
    plt.plot(list_x.ravel()) # flattened array
    plt.title("WithoutNormalized  -- EEG Epoch (30 seconds) (file:"+ path+ ")(#" + str(ramdom_sample)+ ")")
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
  
  
    fig_2 = plt.figure(figsize=(17, 6))
    normalized = rescale_array(list_x)
    plt.plot(normalized.ravel()) # flattened array
    plt.title("Normalized  -- EEG Epoch (30 seconds) (file:"+ path+ ")(#" + str(ramdom_sample)+ ")")
    plt.ylabel("Amplitude")
    plt.xlabel("Time")

    fig_3 = plt.figure(figsize=(17, 6))
    plt.plot(y.ravel()) #flattened array
    plt.title("Sleep Stages (file:"+path+")")
    plt.ylabel("Classes")
    plt.xlabel("Time")
  
  
    print_s('Selected Channel',data['ch_label'] )
    print_s('Sampling Rate',data['fs'] )
    print_s('Points',data['x'] )
    print_s('Hypgnogram',data['y'])

def getIdPerson(path):
    return path.split("/")[-1][:5]
    
def get_data(BASE_PATH):
    # Gets and sorts paths from directory
    files = sorted(glob(os.path.join(BASE_PATH, "*.npz")))

    #example test
    #print("path id person: ", files[0]);
    #print('example id person: ', getIdPerson(files[0]));

    subjects = sorted(list([getIdPerson(x) for x in files]))

    train_ids, test_ids = train_test_split(subjects, test_size=0.2, random_state=1338)

    train_group= [x for x in files if getIdPerson(x) in train_ids]
    test_subjects = [x for x in files if getIdPerson(x) in test_ids]

    print('TOTAL: ' + str(len(test_subjects) +len(train_group) ))
    print('TEST_SUBJECTS: '+str(len(test_subjects)))

    train_subjects, val_subjects = train_test_split(train_group, test_size=0.4, random_state=1338)

    print('TRAIN_SUBJECTS: '+str(len(train_subjects)))
    print('VAL_SUBJECTS: '+str(len(val_subjects)))
    print('____________________________________________')

    train_dict = {k: np.load(k) for k in train_subjects}
    test_dict = {k: np.load(k) for k in test_subjects}
    val_dict = {k: np.load(k) for k in val_subjects}
    
    return train_dict, test_dict, val_dict

def save_plots_by_model(history, path_plot_loss, path_plot_accuracy, accLabel="acc"):
    rain_loss = history['loss']
    val_loss   = history['val_loss']
    train_acc  = history[accLabel]
    val_acc    = history['val_'+accLabel]

    fig_h = plt.figure(figsize=(13, 6))
    plt.plot(rain_loss) 
    plt.plot(val_loss) 
    plt.title("loss vs val_loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(path_plot_loss)

    fig_h2 = plt.figure(figsize=(13, 6))

    plt.plot(train_acc) 
    plt.plot(val_acc) 
    plt.title("acc vs val_acc")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(path_plot_accuracy)

    #(muestras, 100, 3000, 1) 
    # (muestras , 100,5)
    
def print_metrics(model, test_dict, namefile, LENGTH_BASE):
    with open(namefile+'.txt', 'w') as f:
        preds = []
        gt = []

        for record in tqdm(test_dict):
            all_rows = test_dict[record]['x']
            for batch_hyp in chunker(range(all_rows.shape[0])):
                X = all_rows[min(batch_hyp):max(batch_hyp)+1, ...]
                Y = test_dict[record]['y'][min(batch_hyp):max(batch_hyp)+1]
                X = np.resize(X,(X.shape[0], LENGTH_BASE, 1))
                X = np.expand_dims(X, 0)
                X = rescale_array(X)

                Y_pred = model.predict(X)
                Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()

                gt += Y.ravel().tolist()
                preds += Y_pred
        print("_____________________",file = f)
        print(gt,file = f)
        print("_____________________",file = f)
        print(preds,file = f)
        f1 = f1_score(gt, preds, average="macro")
        print("_____________________",file = f)
        print("Seq Test f1 score : %s "% f1,file = f)
        print("_____________________",file = f)
        acc = accuracy_score(gt, preds)

        print("Seq Test accuracy score : %s "% acc,file = f)
        print("_____________________",file = f)
        print(classification_report(gt, preds),file = f)
        print("_____________________",file = f)
        print(confusion_matrix(gt, preds),file = f)
        
        
def training(model, DATA_PATH, RESULT_PATH, LENGTH_BASE, monitorModel="acc",epocs=30, stepsPerEpoch=1000):
    monitorModelVal="val_"+monitorModel
    train_dict, test_dict, val_dict = get_data(DATA_PATH)
    # checkpoint
    filepath=RESULT_PATH+"wi-{epoch:02d}-{"+monitorModelVal+":.2f}.hdf5"
    early = EarlyStopping(monitor=monitorModelVal, mode="max", patience=40, verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor=monitorModelVal, verbose=1, save_best_only=True, mode='max')
    redonplat = ReduceLROnPlateau(monitor=monitorModelVal, mode="max", patience=5, verbose=2)
    callbacks_list = [checkpoint, early]

    print("validation_steps", LENGTH_BASE/BATCH_SIZE)
    validationSteps = int(LENGTH_BASE/BATCH_SIZE);
    # fit generator
    model_history = model.fit_generator(
        gen(train_dict, False, LENGTH_BASE), 
        validation_data=gen(val_dict, False, LENGTH_BASE),
        epochs=epocs, 
        verbose=1, 
        steps_per_epoch=stepsPerEpoch,
        validation_steps = validationSteps,
        callbacks=callbacks_list)

    # save final model
    filepath=RESULT_PATH+"final_model.hdf5"
    model.save(filepath)
    save_plots_by_model(model_history.history, RESULT_PATH+'plot_loss.png', RESULT_PATH+'plot_accuracy.png', monitorModel)

    print_metrics(model, test_dict, RESULT_PATH+"metrics", LENGTH_BASE)
    
print("hola!") 