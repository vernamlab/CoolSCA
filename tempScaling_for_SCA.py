import tensorflow.keras as tk
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Activation
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops.losses import util as tf_losses_utils

import kerastuner as kt
from kerastuner.tuners import *
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from scipy.optimize import minimize
import numpy as np
import sys
import h5py
import numpy as np
from scipy import stats
import scipy.stats as ss
import random
import math
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])
# Defining custom objects before loading trained model

def custom_loss(y_true, y_pred):
    return tk.backend.categorical_crossentropy(y_true[:, :classes], y_pred)


class acc_Metric(tk.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(acc_Metric, self).__init__(name=name, **kwargs)
        self.m = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.m.update_state(K.equal(K.argmax(y_true[:, :classes], axis=-1), K.argmax(y_pred, axis=-1)))

    def result(self):
        return self.m.result()

    def reset_states(self):
        self.m.reset_states()


class Lm_Metric(tk.metrics.Metric):
    def __init__(self, name='lm', **kwargs):
        super(Lm_Metric, self).__init__(name=name, **kwargs)
        self.acc_sum = self.add_weight(name='acc_sum', shape=(256), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc_sum.assign_add(tf_calculate_key_prob(y_true, y_pred))

    def result(self):
        return tf.numpy_function(calculate_Lm, [self.acc_sum], tf.float32)

    def reset_states(self):
        self.acc_sum.assign(K.zeros(256))


class key_rank_Metric(tk.metrics.Metric):
    def __init__(self, name='key_rank', **kwargs):
        super(key_rank_Metric, self).__init__(name=name, **kwargs)
        self.acc_sum = self.add_weight(
            name='acc_sum', shape=(256), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc_sum.assign_add(tf_calculate_key_prob(y_true, y_pred))

    def result(self):
        return tf.numpy_function(rk_key, [self.acc_sum, correct_key], tf.float32)

    def reset_states(self):
        self.acc_sum.assign(K.zeros(256))





def stable_softmax(x):
    z = x - tf.reduce_max(x, axis=-1, keepdims=True)
    numerator = tf.exp(z)
    denominator = tf.reduce_sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax

def no_softmax (x):
    return x


def rank_compute_m(prediction, att_plt, byte, output_rank , mode):
    hw = [bin(x).count("1") for x in range(256)]
    (nb_traces, nb_hyp) = prediction.shape

    key_log_prob = np.zeros(256)
    if mode == "cl":
        prediction = tf.nn.log_softmax((prediction/temp)+1e-40).numpy()
    elif mode== "l":
        prediction = tf.nn.log_softmax((prediction)+1e-40).numpy()
    elif mode == "cs":
        prediction = np.log(stable_softmax((prediction/temp)+1e-40))
    else:
        prediction = np.log(stable_softmax((prediction+1e-40)))


    

    rank_evol = np.full(nb_traces,255)

    for i in range(nb_traces):
        for k in range(256):
            if leakage == 'ID':
                key_log_prob[k] += prediction[i, AES_Sbox[k ^ int(att_plt[i, byte])]]
            else:
                key_log_prob[k] += prediction[i, hw[AES_Sbox[k ^ int(att_plt[i, byte])]]]
        rank_evol[i] = rk_key(key_log_prob, correct_key)

    if output_rank:
        return rank_evol
    else:
        return key_log_prob


def perform_attacks_m(nb_traces, predictions, plt_attack, nb_attacks=1, byte=2, shuffle=True, output_rank=False, mode= 'cl'):
    (nb_total, nb_hyp) = predictions.shape
    all_rk_evol = np.zeros((nb_attacks, nb_traces))

    for i in range(nb_attacks):
        if shuffle:
            l = list(zip(predictions, plt_attack))
            random.shuffle(l)
            sp, splt = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces]
            att_plt = splt[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt_attack[:nb_traces]

        key_log_prob = rank_compute_m(att_pred, att_plt, byte, output_rank , mode)
        if output_rank:
            all_rk_evol[i] = key_log_prob

    if output_rank:
        return np.mean(all_rk_evol,axis=0)  
    else:
        return np.float32(key_log_prob)
    

class TemperatureScaling():
    
    def __init__(self, temp=1, maxiter=50, solver="L-BFGS-B"):

        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, logits, true):

            # Calculates the loss using log-loss (cross-entropy loss)
            scaled_probs = self.predict(logits, x)    
            loss = CategoricalCrossentropy()(tf.convert_to_tensor(true), scaled_probs)
            return loss.numpy()

    # Find the temperature
    def fit(self, logits, true):

        if isinstance(true, tf.Tensor):
            true = true.numpy()
        
        true = true.reshape(-1, true.shape[-1])  # Assuming each row is a one-hot vector
        bounds = [(1e-5, None)]
        opt = minimize(self._loss_fun, x0=1, args=(logits, true), bounds =bounds ,options={'maxiter':self.maxiter}, method=self.solver)
        self.temp = opt.x[0]
        
        return opt
        
    def predict(self, logits, temp=None):

        if temp is None:
            return softmax(logits / self.temp)
        else:
            return softmax(logits / temp)



if __name__ == "__main__":
    #directories to the trained model and traces

    trained_model = ''          
    validation_traces = ''      
    attack_traces=''            
    validation_labels= ''   
    plt_attack = ''    
    validation1 = 12500         


    #load validation traces and the trained model
    model = load_model(trained_model,custom_objects = {"Lm_Metric": Lm_Metric , "key_rank_Metric":key_rank_Metric , "custom_loss":custom_loss})
    X_validation = pd.read_csv(validation_traces , header=None).to_numpy
    X_attack = pd.read_csv(attack_traces , header=None).to_numpy
    Y_validation = pd.read_csv(validation_labels , header=None).to_numpy
    Y_validation = pd.read_csv(plt_attack , header=None).to_numpy





    # getting the raw logits

    x = model.layers[-1].input

    no_softmax = Activation(no_softmax)

    no_softmax_output = no_softmax(x)

    model_No_s = Model(inputs=model.input, outputs=no_softmax_output)

    model_No_s.set_weights(model.get_weights())


    #first validation for temp scaling
    logits = model_No_s.predict(X_validation[:validation1])
    true_labels = Y_validation[:validation1]

    temperature_scaler = TemperatureScaling()
    opt_result = temperature_scaler.fit(logits, true_labels)
    scaled_probs_stable = temperature_scaler.predict(logits)

    temp = opt_result.x[0]
    print(f'temp2 before calibration={temp}')
    #second validation for temp scaling
    logits = model_No_s.predict((X_validation[12500:]))
    true_labels = Y_validation[12500:]

    temperature_scaler = TemperatureScaling()
    opt_result = temperature_scaler.fit(logits/temp, true_labels)
    scaled_probs_stable = temperature_scaler.predict(logits)
    print(f'temp2 after calibration={opt_result.x[0]}')




    #attack phase
    predictions = model_No_s(X_attack)

    avg_rank_cl = np.array(perform_attacks_m(5000, predictions, plt_attack, nb_attacks=10, byte=2, shuffle=True, output_rank=True , mode='cl'))
    avg_rank_l = np.array(perform_attacks_m(5000, predictions, plt_attack, nb_attacks=10, byte=2, shuffle=True, output_rank=True , mode='l'))
    avg_rank_cs = np.array(perform_attacks_m(5000, predictions, plt_attack, nb_attacks=10, byte=2, shuffle=True, output_rank=True , mode='cs'))
    avg_rank_s = np.array(perform_attacks_m(5000, predictions, plt_attack, nb_attacks=10, byte=2, shuffle=True, output_rank=True , mode='s'))




    #plot rank curves

    cutoff = 5000
    x = np.linspace(0, cutoff, cutoff)

    plt.figure(figsize=(8, 6))



    plt.plot(x, avg_rank_cl[:cutoff], color='#DC143C', label='Calibrated Softmax', linewidth=2)
    plt.plot(x, avg_rank_l[:cutoff], color='#FF6347', label='Softmax', linewidth=2, marker='s', markersize=3, markevery=100)
    plt.plot(x, avg_rank_cs[:cutoff], color='#4169E1', label='Calibrated Stable Softmax', linewidth=2)
    plt.plot(x, avg_rank_s[:cutoff], color='#00BFFF', label='Stable Softmax', linewidth=2, marker='s', markersize=3, markevery=100)

    
    plt.xlabel('Attack Traces', fontsize=16)
    plt.ylabel('Guessing Entropy', fontsize=16)

    
    plt.ylim(0, 256)
    plt.xlim(left=0, right=cutoff) 

    plt.xticks([0] + list(np.arange(500, cutoff+1, 500)), fontsize=10)  
    plt.yticks(list(np.arange(50, 257, 50)), fontsize=10)            

    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.legend(fontsize=16)

    plt.margins(x=0, y=0)

    ax = plt.gca()


    plt.show()
