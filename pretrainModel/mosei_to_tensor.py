import mmsdk
from mmsdk import mmdatasdk as md
import numpy as np
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook
from collections import defaultdict
from tensor_fusion import TensorFusionNetwork
# remove later
# from tensor_fusion import TFN
# from torch.autograd import Variable

def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch])
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    
    # return sentences.permute(1,0,2), visual.permute(1,0,2), acoustic.permute(1,0,2), labels, lengths
    return sentences, visual, acoustic, labels, lengths

# def multi_collate(batch):
#     """
#     Collate function that processes the data batch to output fusion_tensor along with labels and lengths.
#     Assume batch contains list of tuples: [(text_tensor, video_tensor, audio_tensor, label, length), ...]
#     """
#     batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)

#     # Convert lists to tensors
#     labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0).float()
#     sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch]).float().permute(1,0,2)
#     visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch]).permute(1,0,2)
#     acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch]).permute(1,0,2)

#     lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch]).float()
    

#     # Create an instance of your fusion network
#     # Assuming you have defined dimensions previously or have them set globally
#     tfn = TensorFusionNetwork(text_dim=300, visual_dim=430, acoustic_dim=74, fusion_dim=128)
    
#     # Forward pass through your TensorFusionNetwork
#     fusion_tensors = tfn(sentences, visual, acoustic, lengths)

#     # Return the new batch format
#     return fusion_tensors, labels, lengths


# function that returns train_set, valid_set, and test_set
def preprocess_mosi():
    # Preprocessing CMU-MOSI

    DATASET = md.cmu_mosei

    text_size = 0
    visual_size = 0
    acoustic_size = 0

    # define your different modalities - refer to the filenames of the CSD files
    text_field = 'CMU_MOSEI_TimestampedWordVectors' # 300-dimensional GloVe word vecs
    acoustic_field = 'CMU_MOSEI_COVAREP'
    # visual_field = 'CMU_MOSEI_VisualFacet42'
    visual_field = 'CMU_MOSEI_VisualOpenFace2'
    label_field = 'CMU_MOSEI_Labels'

    # creating a dataset containing the aligned version of mosi 
    features = [text_field, acoustic_field, visual_field, label_field]

    recipe = {feat: os.path.join('../mosei_final_aligned', feat) + '.csd' for feat in features}
    aligned_mosi = md.mmdataset(recipe)

    # Split preprocessed data to train, validation, and test 
    trainset = DATASET.standard_folds.standard_train_fold
    validationset = DATASET.standard_folds.standard_valid_fold
    testset = DATASET.standard_folds.standard_test_fold

    # print(trainset)

    # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
    EPS = 0

    # place holders for the final train/dev/test dataset
    train = []
    valid = []
    test = []

    # define a regular expression to extract the video ID out of the keys
    pattern = re.compile('(.*)\[.*\]')
    num_drop = 0 # a counter to count how many data points went into some processing issues

    for segment in aligned_mosi[label_field].keys():
        
        # get the video ID and the features out of the aligned dataset
        vid = re.search(pattern, segment).group(1)
        label = aligned_mosi[label_field][segment]['features']
        words = aligned_mosi[text_field][segment]['features']
        visual = aligned_mosi[visual_field][segment]['features']
        acoustic = aligned_mosi[acoustic_field][segment]['features']

        # if the sequences are not same length after alignment, there must be some problem with some modalities
        # we should drop it or inspect the data again
        if not words.shape[0] == visual.shape[0] == acoustic.shape[0]:
            print(f"Encountered datapoint {vid} with text shape {words.shape}, visual shape {visual.shape}, acoustic shape {acoustic.shape}")
            num_drop += 1
            continue

        # remove nan values
        label = np.nan_to_num(label)
        words = np.nan_to_num(words)
        visual = np.nan_to_num(visual)
        acoustic = np.nan_to_num(acoustic)

        words = np.asarray(words)
        visual = np.asarray(visual)
        acoustic = np.asarray(acoustic)

        # z-normalization per instance and remove nan/infs
        visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
        acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

        if vid in trainset:
            train.append(((words, visual, acoustic), label, segment))
        elif vid in validationset:
            valid.append(((words, visual, acoustic), label, segment))
        elif vid in testset:
            test.append(((words, visual, acoustic), label, segment))
        else:
            print(f"Found video that doesn't belong to any splits: {vid}")
        
    print(f"Total number of {num_drop} datapoints have been dropped.")

    return train, valid, test

# preprocess_mosi()
# train, valid, test = preprocess_mosi()
# batch_sz = 56
# train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate)
# valid_loader = DataLoader(valid, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)
# test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)
# for i in train_loader:
#     print(i[0].shape)
#     print(i[1].shape)
#     print(i[2].shape)
#     break