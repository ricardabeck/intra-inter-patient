"""
    conda deactivate
To run in background:
    
    source venv_tf/bin/activate 
    cd diffpriv/
    nohup python -m 04b_interpatient_shadowmodel > logs/interpatient_shadow.log 2>&1

To check resource consumption:
    ps ux --sort=%cpu

Runtime:


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import tensorflow_addons as tfa
from os.path import join as osj

from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Embedding, Dense, Bidirectional, Input
from tensorflow.keras import Model

import random
import pickle
import time
import os
import argparse

from datetime import datetime
from sklearn.metrics import confusion_matrix
from  sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

random.seed(654)

def read_inter_attack_setups():
    inter_setups = pd.read_csv('../results_dp/attack_setup_inter.csv')
    return inter_setups

def get_attack_setups():
    """
    Loads the attack setups for intra and inter patient attacks.

    Returns:
        dict_all_setups: dict with structure {model: {epsilon: [delta]}}
    """

    # inter
    inter_setups = read_inter_attack_setups()
    inter_setups = inter_setups.sort_values(by=["Model", "Epsilon", "Delta"], ascending=False)
    inter_setups["Model"] = inter_setups["Model"].str.replace("Inter-", "", regex=False)

    # all
    inter_setups = inter_setups.sort_values(by=["Model", "Epsilon", "Delta"], ascending=False)

    # dict
    dict_all_setups = {}
    for _, row in inter_setups.iterrows():
        model = row["Model"]
        epsilon = row["Epsilon"]
        delta = row["Delta"]
        
        if model not in dict_all_setups:
            dict_all_setups[model] = {}
        if epsilon not in dict_all_setups[model]:
            dict_all_setups[model][epsilon] = []
        
        dict_all_setups[model][epsilon].append(delta)

    return dict_all_setups

def read_dp_signals(m, e):
    with open(osj("..", "data_dp", f"{m}_{e}.pkl"), "rb") as f:
        return pickle.load(f)

def read_results():
    with open(osj("..", "results_attacks", "interpatient_results.pkl"), "rb") as f:
        results =  pickle.load(f)
    with open(osj("..", "results_attacks", "interpatient_loss.pkl"), "rb") as f:
        loss =  pickle.load(f)
    with open(osj("..", "results_attacks", "interpatient_probs.pkl"), "rb") as f:
        probs =  pickle.load(f)
    return results, loss, probs

# F1 Read files
def read_data(filename, values, max_time=100, classes=['N', 'S', 'V'], max_label=100, trainset=1):

    random.seed(654)
    beats = [] 
    dict_samples = spio.loadmat(filename + '.mat')
    samples = dict_samples['s2s_mitbih']
    labels = samples[0]['seg_labels']

    # patient IDs for original train / test set
    # Train set: DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118,119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223,230];
    # Test set: DS2 = [100, 103, 105, 111, 113, 117, 121, 123,200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233,234];
    # assuming the test set was leaked or is publicly available
    #DS1_idx = [1, 6, 8,  9, 11, 13, 14, 15, 17, 18, 20, 22, 24, 26, 27, 28, 29, 30, 35, 38, 41, 43]
    DS2_shadow_train = [0,  5, 12, 19, 23, 31, 33, 37, 44, 45, 46]
    DS2_shadow_test  = [3, 10, 16, 21, 25, 32, 34, 39, 40, 42, 47]

    # Select train and test data
    if trainset == 1:
        temp_values = [values[i] for i in DS2_shadow_train]
        labels = [labels[i] for i in DS2_shadow_train]

    elif trainset == 0:
        temp_values = [values[i] for i in DS2_shadow_test]
        labels = [labels[i] for i in DS2_shadow_test]

    # calculate the number of annotations and sequences
    num_annots = sum([item.shape[0] for item in temp_values])
    n_seqs = num_annots / max_time 

    # add all beats together
    count_b = 0
    nr_recordings = [] # number of recordings per patient (each recording contains 280 measurements)
    for _, item in enumerate(temp_values):
        l = item.shape[0] # number of recordings per patient (each recording contains 280 measurements)
        nr_recordings.append(l)
        for itm in item:
            if count_b == num_annots: # hence all recordings have been added
                break
            beats.append(itm[0]) # itm is one recording, with 280 measurements
            count_b += 1

    # add all labels together
    count_l  = 0
    t_labels = []
    for _, item in enumerate(labels): 
        if len(t_labels) == num_annots: # break if all labels have been added
            break
        item = item[0]
        # iterate over all recordings per patient
        for lbl in item: 
            if count_l == num_annots: # break if all labels have been added
                break
            t_labels.append(str(lbl))
            count_l += 1
    
    del temp_values
    # convert list to array & reshape
    beats = np.asarray(beats)
    t_labels = np.asarray(t_labels)  
    shape_v = beats.shape
    beats = np.reshape(beats, [shape_v[0], -1])

    # Create empty arrays for data and labels
    random_beats  = np.asarray([],dtype=np.float64).reshape(0,shape_v[1])
    random_labels = np.asarray([],dtype=np.dtype('|S1')).reshape(0,)

    # iterate over all classes and truncate to max_label samples, so that all classes are equally represented
    for cl in classes:
        _label = np.where(t_labels == cl) # select indices that match the class
        logger.info(f"Class {cl} is represented {len(_label[0])}")

        # random permutation of indices
        permute = np.random.permutation(len(_label[0])) 
        _label = _label[0][permute[:max_label]] # choose the first X indices
        logger.info(f"Class {cl} is now represented {len(_label)}")

        random_beats = np.concatenate((random_beats, beats[_label]))
        random_labels = np.concatenate((random_labels, t_labels[_label]))

    # shorten data to multiple of max_time
    signals = random_beats[:int(len(random_beats)/ max_time) * max_time, :]
    _labels  = random_labels[:int(len(random_beats) / max_time) * max_time]

    #  reshape data into groups of max_time
    data   = [signals[i:i + max_time] for i in range(0, len(signals), max_time)]
    labels = [_labels[i:i + max_time] for i in range(0, len(_labels), max_time)]

    permute = np.random.permutation(len(labels)) # random permutation of indices only

    # transform from list to array
    data   = np.asarray(data, dtype=object) 
    labels = np.asarray(labels, dtype=object)

    # reorder data and labels according to random permute
    data   = data[permute]
    labels = labels[permute]

    logger.info('Signals and labels processed!')

    return data, labels

# F2 Normaliza data
def normalize(data):
        data = np.nan_to_num(data)  # removing NaNs and Infs
        data = data - np.mean(data)
        data = data / np.std(data)
        return data

# F3 shuffle
def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    #     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start + batch_size], y[start:start + batch_size]
        start += batch_size

# F4 bool value check
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# F5 calculate performance
def evaluate_metrics(confusion_matrix):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(ACC) # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    # F1 Score (Harmonic Mean of Precision & Recall)
    F1 = 2 * (PPV * TPR) / (PPV + TPR)

    return ACC_macro, ACC, TPR, TNR, PPV, NPV, FPR, FNR, FDR, F1

# F6 Network
def build_network(inputs, dec_inputs, char2numY, n_channels=10, input_depth=280, num_units=128, max_time=10, bidirectional=False):
    # Reshape the inputs to match the Conv1D expected shape
    _inputs = tf.reshape(inputs, [-1, n_channels, int(input_depth / n_channels)])
    
    # Convolutional and MaxPooling layers
    conv1 = Conv1D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu')(_inputs)
    max_pool_1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(conv1)

    conv2 = Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu')(max_pool_1)
    max_pool_2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(conv2)

    conv3 = Conv1D(filters=128, kernel_size=2, strides=1, padding='same', activation='relu')(max_pool_2)

    # Flatten the output of the Conv1D layers
    shape = conv3.shape.as_list()
    data_input_embed = tf.reshape(conv3, (-1, max_time, shape[1] * shape[2]))

    # Embedding for the decoder
    embed_size = 10
    output_embedding = tf.Variable(tf.random.uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
    data_output_embed = tf.nn.embedding_lookup(params=output_embedding, ids=dec_inputs)

    # Encoder
    if not bidirectional:
        # Regular LSTM
        lstm_enc = LSTM(num_units, return_state=True)
        _, last_state_h, last_state_c = lstm_enc(data_input_embed)
        last_state = (last_state_c, last_state_h)
    else:
        # Bidirectional LSTM
        lstm_enc = Bidirectional(LSTM(num_units, return_state=True))
        _, forward_h, forward_c, backward_h, backward_c = lstm_enc(data_input_embed)
        last_state_c = tf.concat([forward_c, backward_c], axis=-1)
        last_state_h = tf.concat([forward_h, backward_h], axis=-1)
        last_state = (last_state_c, last_state_h)

    # Decoder
    if not bidirectional:
        lstm_dec = LSTM(num_units, return_sequences=True)
    else:
        lstm_dec = LSTM(2 * num_units, return_sequences=True)

    dec_outputs = lstm_dec(data_output_embed, initial_state=last_state)

    # Final Dense layer to produce logits
    logits = Dense(len(char2numY))(dec_outputs)

    return logits

# 7 Model evaluation
def test_model(sess, logits, X_test, y_test, batch_size, char2numY, y_seq_length, inputs, dec_inputs):
    # source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))
    acc_track = []
    sum_test_conf = []
    count = 0

    all_probs = [] 
    all_true_labels = []
    all_pred_labels = []

    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_test, y_test, batch_size)):
        #logger.info(f"Running batch: {count}")
        count += 1
        dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
        batch_probs = []

        for i in range(y_seq_length):
            batch_logits = sess.run(logits, feed_dict={inputs: source_batch, dec_inputs: dec_input})
            softmax_probs = sess.run(tf.nn.softmax(batch_logits[:, -1]), feed_dict={}) 
            prediction = batch_logits[:, -1].argmax(axis=-1)

            batch_probs.append(softmax_probs)
            dec_input = np.hstack([dec_input, prediction[:, None]])

        acc_track.append(dec_input[:, 1:] == target_batch[:, 1:])
        y_true= target_batch[:, 1:].flatten()
        y_pred = dec_input[:, 1:].flatten()

        sum_test_conf.append(confusion_matrix(y_true, y_pred,labels=list(range(len(char2numY)-1))))

        all_probs.append(np.stack(batch_probs, axis=1)) # shape: [batch, time, n_classes]
        all_true_labels.append(target_batch[:, 1:])
        all_pred_labels.append(dec_input[:, 1:])

    sum_test_conf= np.mean(np.array(sum_test_conf, dtype=np.float32), axis=0)
    acc_avg, acc, sensitivity, specificity, ppv, npv, fpr, fnr, fdr, f1_score = evaluate_metrics(sum_test_conf)

    logger.info(f"Average Accuracy is: {acc_avg} on test set")

    # for idx in range(n_classes):
        # logger.info(f"\t{classes[idx]} rhythm -> Sensitivity: {sensitivity[idx]}, Specificity : {specificity[idx]}, Precision (PPV) : {ppv[idx]}, Accuracy : {acc[idx]}")

    logger.info(f"\t Average -> Sensitivity: {np.mean(sensitivity)}, Specificity : {np.mean(specificity)}, Precision (PPV) : {np.mean(ppv)}, Accuracy : {np.mean(acc)}")

    all_probs = np.concatenate(all_probs, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_pred_labels = np.concatenate(all_pred_labels, axis=0)
    
    return acc_avg, acc, sensitivity, specificity, ppv, npv, fpr, fnr, fdr, f1_score, all_probs, all_true_labels, all_pred_labels

def initialize_results_loss(dict_attacks):

    all_results = {mechanism: {epsilon: {delta: None for delta in dict_attacks[mechanism][epsilon]} for epsilon in dict_attacks[mechanism].keys()} for mechanism in dict_attacks.keys()}
    loss_track = {mechanism: {epsilon: {delta: None for delta in dict_attacks[mechanism][epsilon]} for epsilon in dict_attacks[mechanism].keys()} for mechanism in dict_attacks.keys()}
    probs = {mechanism: {epsilon: {delta: None for delta in dict_attacks[mechanism][epsilon]} for epsilon in dict_attacks[mechanism].keys()} for mechanism in dict_attacks.keys()}

    return all_results, loss_track, probs

################
# MAIN PROCESS #
################

def run_program(args):
    
    # Arguments
    logger.info(args)
    max_time = args.max_time # 5 3 second best 10# 40 # 100
    epochs = args.epochs # 300
    batch_size = args.batch_size # 10
    num_units = args.num_units
    bidirectional = args.bidirectional
    # lstm_layers = args.lstm_layers
    n_oversampling = args.n_oversampling
    checkpoint_dir = args.checkpoint_dir
    ckpt_name = args.ckpt_name
    test_steps = args.test_steps
    classes= args.classes
    filename = args.data_dir
    n_channels = 10

    # get attack combinations
    dict_attacks = get_attack_setups()
    last_epsilon = 0.0

    # check for existing results, to skip already trained models
    if os.path.exists(osj("..", "results_attacks", "interpatient_results.pkl")):
        all_results, loss_track, all_probs_output = read_results()
    else:
        all_results, loss_track, all_probs_output = initialize_results_loss(dict_attacks)

    for mechanism  in dict_attacks.keys():
        last_epsilon = 0.0

        for epsilon in dict_attacks[mechanism].keys():

            # select epsilon for loading the correct file
            if epsilon <= 0.091:
                file_epsilon = 0.091
            elif epsilon <= 0.91:
                file_epsilon = 0.91
            elif epsilon <= 2.01:
                file_epsilon = 2.01

            if last_epsilon != file_epsilon:
                # Load DP signals
                logger.info(f"Loading DP signals file for {mechanism} until epsilon {file_epsilon} ...")
                dp_signals = read_dp_signals(mechanism, file_epsilon)
            
            last_epsilon = file_epsilon

            for delta in dict_attacks[mechanism][epsilon]:

                if all_results[mechanism][epsilon][delta] is not None:
                    logger.info(f"Skipping existing delta {delta} for epsilon {epsilon}...")
                    skipped = True
                    continue
                # elif dp_signals[epsilon][delta] is None:
                #     logger.info(f"Skipping empty delta {delta} for epsilon {epsilon}...")
                #     skipped = True
                #     continue
                else:
                    skipped = False
                    logger.info(f"Processing mechanism: {mechanism}, epsilon: {epsilon}, delta: {delta}")

                    # specifying a new directory to save the checkpoint per epsilon and delta combination 
                    checkpoint_dir_temp = f"{checkpoint_dir}_{mechanism}_e{epsilon}_d{delta}"

                    # STEP 1 Read data
                    X_train, y_train = read_data(filename, dp_signals[epsilon][delta], max_time, classes=classes, max_label=50000,trainset=1)
                    X_test, y_test = read_data(filename, dp_signals[epsilon][delta], max_time, classes=classes, max_label=50000,trainset=0)          

                    input_depth = X_train.shape[2]
                    classes = np.unique(y_train)
                    char2numY = dict(list(zip(classes, list(range(len(classes))))))
                    n_classes = len(classes)

                    for cl in classes:
                        ind = np.where(classes == cl)[0][0]
                        # logger.info(f"Class: {cl} - count: {len(np.where(Y.flatten() == cl)[0])}")

                    char2numY['<GO>'] = len(char2numY)
                    num2charY = dict(list(zip(list(char2numY.values()), list(char2numY.keys()))))

                    y_train = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y_train]
                    y_test  = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y_test]
                    y_test  = np.asarray(y_test)
                    y_train = np.array(y_train)

                    x_seq_length = len(X_train[0])
                    y_seq_length = len(y_train[0])- 1

                    # Placeholders
                    tf.compat.v1.disable_eager_execution()
                    inputs = tf.compat.v1.placeholder(tf.float32, [None, max_time, input_depth], name = 'inputs')
                    targets = tf.compat.v1.placeholder(tf.int32, (None, None), 'targets')
                    dec_inputs = tf.compat.v1.placeholder(tf.int32, (None, None), 'output')

                    logits = build_network(inputs, dec_inputs, char2numY, n_channels=n_channels, input_depth=input_depth, num_units=num_units, max_time=max_time,
                                bidirectional=bidirectional)
                    
                    with tf.compat.v1.name_scope("optimization"):
                        # Loss function
                        vars = tf.compat.v1.trainable_variables()
                        beta = 0.001
                        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * beta
                        loss = tfa.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
                        loss = tf.reduce_mean(input_tensor=loss + lossL2)
                        # Optimizer
                        optimizer = tf.compat.v1.train.RMSPropOptimizer(1e-3).minimize(loss)

                    # over-sampling: SMOTE
                    X_train = np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],-1])
                    y_train= y_train[:,1:].flatten()

                    nums = []
                    for cl in classes:
                        ind = np.where(classes == cl)[0][0]
                        nums.append(len(np.where(y_train.flatten()==ind)[0]))

                    ratio={0:nums[0],1:n_oversampling+1000,2:n_oversampling*2}
                    sm = SMOTE(random_state=12,sampling_strategy=ratio)

                    X_train, y_train = sm.fit_resample(X_train, y_train)

                    X_train = X_train[:int(X_train.shape[0]/max_time)*max_time,:]
                    y_train = y_train[:int(X_train.shape[0]/max_time)*max_time]

                    X_train = np.reshape(X_train,[-1,X_test.shape[1],X_test.shape[2]])
                    y_train = np.reshape(y_train,[-1,y_test.shape[1]-1,])

                    y_train= [[char2numY['<GO>']] + [y_ for y_ in date] for date in y_train]
                    y_train = np.array(y_train)

                    if (os.path.exists(checkpoint_dir_temp) == False):
                        os.mkdir(checkpoint_dir_temp)
                    
                    epoch_var = tf.Variable(0, dtype=tf.int32, trainable=False, name="epoch")
                    
                    # train the graph
                    with tf.compat.v1.Session() as sess:
                        sess.run(tf.compat.v1.global_variables_initializer())
                        sess.run(tf.compat.v1.local_variables_initializer())
                        train_probs = []
                        train_true = []
                        train_pred = [] 

                        saver = tf.compat.v1.train.Saver()
                        # saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
                        ckpt = tf.train.get_checkpoint_state(checkpoint_dir_temp)

                        epoch_loss = {}
                        epoch_results = {}
                        
                        for epoch_i in range(epochs):
                            test_results = {}
                            start_time = time.time()
                            train_acc = []
                            batches_loss = {}
                            for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
                                _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                                    feed_dict = {inputs: source_batch,
                                                    dec_inputs: target_batch[:, :-1],
                                                    targets: target_batch[:, 1:]})

                                train_probs.append(batch_logits)  # [B, T, C]
                                train_true.append(target_batch[:, 1:])  # [B, T]
                                train_pred.append(batch_logits.argmax(axis=-1))  # [B, T]
                                batches_loss[batch_i] = batch_loss
                                train_acc.append(batch_logits.argmax(axis=-1) == target_batch[:,1:])
                            
                            epoch_loss[epoch_i] = batches_loss

                            accuracy = np.mean(train_acc)
                            logger.info(f"Epoch {epoch_i+1} Loss: {batch_loss} Accuracy: {accuracy} Epoch duration: {time.time() - start_time}s ")

                            if (epoch_i+1)%test_steps==0:
                                acc_avg, acc, sensitivity, specificity, ppv, npv, fpr, fnr, fdr, f1_score, test_probs, test_true, test_pred = test_model(sess, logits, X_test, y_test, batch_size, char2numY, y_seq_length, inputs, dec_inputs) # definition moved to function definitions (Ricarda)
                                test_results["avg_acc"] = acc_avg
                                test_results["acc"]     = acc
                                test_results["sens"]    = sensitivity
                                test_results["spec"]    = specificity
                                test_results["prec"]    = ppv
                                test_results["neg_pred_value"] = npv
                                test_results["false_pos_rate"] = fpr
                                test_results["false_neg_rate"] = fnr
                                test_results["false_det_rate"] = fdr
                                test_results["f1_score"]       = f1_score

                                epoch_results[epoch_i] = test_results
                            
                            save_path = os.path.join(checkpoint_dir_temp, ckpt_name)
                            sess.run(tf.compat.v1.assign(epoch_var, epoch_i+1))
                                  
                            saver.save(sess, save_path)
                            logger.info(f"Model saved in path {save_path}")
            
                    
                    all_probs_output[mechanism][epsilon][delta]["train"] = {
                        'probs': train_probs,           
                        'true': train_true,            
                        'pred': train_pred     
                    }

                    all_probs_output[mechanism][epsilon][delta]["test"] = {
                        'probs': test_probs,           
                        'true': test_true,            
                        'pred': test_pred     
                    }
                            
                    all_results[mechanism][epsilon][delta] = epoch_results
                    loss_track[mechanism][epsilon][delta] = epoch_loss


                if skipped == False:
                    with open(osj("..", "results_attacks", "interpatient_results.pkl"), "wb") as f:
                        pickle.dump(all_results, f) 
                    with open(osj("..", "results_attacks", "interpatient_loss.pkl"), "wb") as f:
                        pickle.dump(loss_track, f) 
                    with open(osj("..", "results_attacks", "interpatient_probs.pkl"), "wb") as f:
                        pickle.dump(all_probs_output, f) 
    
    # save again when finished
    with open(osj("..", "results_attacks", "interpatient_results.pkl"), "wb") as f:
        pickle.dump(all_results, f) 
    with open(osj("..", "results_attacks", "interpatient_loss.pkl"), "wb") as f:
        pickle.dump(loss_track, f) 
    with open(osj("..", "results_attacks", "interpatient_probs.pkl"), "wb") as f:
        pickle.dump(all_probs_output, f) 
    
    logger.info(f"Saved results and loss for all combinations.")
    logger.info(f"Model training finished!")


# 6 configure arguments
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_time', type=int, default=10) # originally 10
    parser.add_argument('--test_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20) # originally 20
    parser.add_argument('--bidirectional', type=str2bool, default=str2bool('False')) # originally False
    # parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--num_units', type=int, default=128)
    parser.add_argument('--n_oversampling', type=int, default=10000)
    parser.add_argument('--data_dir', type=str, default='../data/s2s_mitbih_aami')
    parser.add_argument('--checkpoint_dir', type=str, default='dp_checkpoints_DS/shadow/checkpoints')
    parser.add_argument('--ckpt_name', type=str, default='seq2seq.ckpt')
    parser.add_argument('--classes', nargs='+', type=chr, default=['N','S','V']) 
    args = parser.parse_args()
    run_program(args)

if __name__ == '__main__':
    main()