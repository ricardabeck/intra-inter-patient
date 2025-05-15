
"""
To run:

    conda deactivate
    source venv_tf/bin/activate 
    cd diffpriv/
    nohup python -m rundiffpriv_gaussian > logs/rundiffpriv_gaussian.log 2>&1

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import tensorflow_addons as tfa
from os.path import join as osj
import pandas as pd

from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Embedding, Dense, Bidirectional, Input
from tensorflow.keras import Model

import random
import pickle
import time
import os
import argparse
import copy

from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Differential privacy libraries
from diffprivlib import mechanisms
from diffprivlib import models
from diffprivlib import tools
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import check_random_state
from diffprivlib.mechanisms import Laplace, LaplaceBoundedNoise, GaussianAnalytic
from diffprivlib.mechanisms import DPMechanism

from collections import Counter

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

random.seed(42)

#############
# FUNCTIONS #
#############

def get_data():
    dict_samples = spio.loadmat('../data/s2s_mitbih_aami.mat')
    # dict_samples = spio.loadmat('../data/s2s_mitbih_aami_DS1DS2.mat')
    samples = dict_samples['s2s_mitbih'] # 2D array with 2 columns: ecg values and labels
    values = samples[0]['seg_values'] # ecg values
    return values

def get_patient_ids():
    with open(osj("..", "data", "all_patients.pkl"), "rb") as f:
        return pickle.load(f)

def read_dp_signals(m, e):
    with open(osj("..", "data_dp", f"{m}_{e}.pkl"), "rb") as f:
        return pickle.load(f)

def set_dp_mechanism(m, e, d, s): 
    seed = random.seed(42)
    if m == 'laplace':
        dp_mechanism = Laplace(epsilon=e, delta=d, sensitivity=s, random_state=seed)
    elif m == 'bounded_n':
        dp_mechanism = LaplaceBoundedNoise(epsilon=e, delta=d, sensitivity=s, random_state=seed) # Delta must be > 0 and in (0, 0.5).
    elif m == "gaussian_a":
        dp_mechanism = GaussianAnalytic(epsilon=e, delta=d, sensitivity=s, random_state=seed)

    return dp_mechanism


def run_diffpriv(method, epsilon, delta, sensitivity, values):
    random.seed(42) 

    ecgs = copy.deepcopy(values)
    mechanism = set_dp_mechanism(method, epsilon, delta, sensitivity)

    patient_count = 0
    ########  PATIENT  ########
    for patient in values: 
        segment_count = 0
        logger.info(f"Starting with patient {patient_count} ...")

        ########  SEGMENT  ########
        for segment in patient:
            signal_count = 0 

            ########  SIGNAL  ########            
            for signal in ecgs[patient_count][segment_count][0]:
                dp_signal = mechanism.randomise(signal.item())
                ecgs[patient_count][segment_count][0][signal_count] = dp_signal
                signal_count += 1
            
            segment_count += 1
        
        patient_count += 1

    return ecgs

def save_dp_signals(dict_signals_dp, m, e):
    with open(osj("..", "data_dp", f"{m}_{e}.pkl"), "wb") as f:
        pickle.dump(dict_signals_dp, f)


################
# MAIN PROCESS #
################

def apply_diffpriv():

    mechanism = "gaussian_a"
    # hp_epsilon_values = [0.001, 0.01, 0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091] # done
    # hp_epsilon_values = [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91] # done
    hp_epsilon_values = [1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91, 2.01]
    hp_delta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    file_epsilon = hp_epsilon_values[-1]
    sensitivity = 0.49
    values = get_data()

    logger.info(f"Setup for differential privacy with {mechanism} for all patients per epsilon.")
    dict_signals_dp = dict.fromkeys(hp_epsilon_values)

    if os.path.exists(osj("..", "data_dp", f"{mechanism}_{file_epsilon}.pkl")):
        dict_signals_dp = read_dp_signals(mechanism, file_epsilon)

    ########  EPSILON  ########
    for epsilon in hp_epsilon_values:
        
        if (dict_signals_dp[epsilon] is not None) and (len(dict_signals_dp[epsilon]) == len(hp_delta_values)):
            logger.info(f"Skipping complete epsilon {epsilon} ...")
            continue   
        else:

            temp_dict_signals = {}

            ########  DELTA  ########
            for delta in hp_delta_values:

                try:
                    dict_signals_dp[epsilon][delta]
                    logger.info(f"Skipping delta {delta} ...")
                    continue

                except (TypeError, KeyError):
                    
                    logger.info(f"Calculating data for epsilon {epsilon} and delta {delta} ...")
                    dp_all_patients = run_diffpriv(mechanism, epsilon, delta, sensitivity, values)     
                    temp_dict_signals[delta] = dp_all_patients
                    
            # save dp signals 
            dict_signals_dp[epsilon] = temp_dict_signals   

    save_dp_signals(dict_signals_dp, mechanism, epsilon)
    logger.info(f"Saved results for epsilon {epsilon} and delta {delta}")


def main():
    apply_diffpriv()

if __name__ == '__main__':
    main()