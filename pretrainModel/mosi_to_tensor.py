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
# remove later
from tensor_fusion import TFN
from torch.autograd import Variable

# loading multiple datapoints from an iterable dataset and putting them into certain format
def tensor_fusion(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    DTYPE = torch.FloatTensor
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=0).permute(1,0,2)
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch]).permute(1,0,2)
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch]).permute(1,0,2)

    # hard code input_dims, hidden_dims for now 
    input_dims = [74, 47, 300]
    hidden_dims = [int(i*1.5) for i in input_dims]

    # initialize TFN to fuse tensors of three modalities
    tfn = TFN(input_dims, hidden_dims, 64, (0.25, 0.25, 0.25, 0.25), 32)
    tfn.zero_grad()

    x_t = Variable(sentences.float().type(DTYPE), requires_grad=False)
    x_v = Variable(visual.float().type(DTYPE), requires_grad=False).squeeze()
    x_a = Variable(acoustic.float().type(DTYPE), requires_grad=False).squeeze()
            
    # y is the labels
    y = Variable(labels.view(-1, 1).float().type(DTYPE), requires_grad=False)
    fused_input = tfn(x_a, x_v, x_t)

    return fused_input, y

# function that returns train_set, valid_set, and test_set
def preprocess_mosi():
    # Preprocessing CMU-MOSI

    DATASET = md.cmu_mosi

    # define your different modalities - refer to the filenames of the CSD files
    text_field = 'CMU_MOSI_TimestampedWordVectors' # 300-dimensional GloVe word vecs
    acoustic_field = 'CMU_MOSI_COVAREP'
    visual_field = 'CMU_MOSI_Visual_OpenFace_1'
    label_field = 'CMU_MOSI_Opinion_Labels'

    # creating a dataset containing the aligned version of mosi 
    features = [text_field, acoustic_field, visual_field, label_field]

    recipe = {feat: os.path.join('./cmumosi_final_aligned', feat) + '.csd' for feat in features}
    aligned_mosi = md.mmdataset(recipe)

    # testing what all the video keys look like
    random_key = list(aligned_mosi[visual_field].keys())[100]
    print('Key: ', random_key)
    print('Interval size: ', aligned_mosi[visual_field][random_key]['intervals'].shape)
    print('Visual features size: ', aligned_mosi[visual_field][random_key]['features'].shape)
    print('Acoustic features size: ', aligned_mosi[acoustic_field][random_key]['features'].shape)
    print('Text feature size: ', aligned_mosi[text_field][random_key]['features'].shape)

    # Split preprocessed data to train, validation, and test 
    trainset = DATASET.standard_folds.standard_train_fold
    validationset = DATASET.standard_folds.standard_valid_fold
    testset = DATASET.standard_folds.standard_test_fold

    # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
    EPS = 0

    # construct a word2id mapping that automatically takes increment when new words are encountered
    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['<unk>']
    PAD = word2id['<pad>']

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
        _words = aligned_mosi[text_field][segment]['features']
        _visual = aligned_mosi[visual_field][segment]['features']
        _acoustic = aligned_mosi[acoustic_field][segment]['features']

        # if the sequences are not same length after alignment, there must be some problem with some modalities
        # we should drop it or inspect the data again
        if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
            print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
            num_drop += 1
            continue

        # remove nan values
        label = np.nan_to_num(label)
        _visual = np.nan_to_num(_visual)
        _acoustic = np.nan_to_num(_acoustic)

        # remove speech pause tokens - this is in general helpful
        # we should remove speech pauses and corresponding visual/acoustic features together
        # otherwise modalities would no longer be aligned
        words = []
        visual = []
        acoustic = []
        # for i, word in enumerate(_words):
        #     if word[0] != b'sp':
        #         # words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
        #         words.append(word2id[str(word[0])])
        #         visual.append(_visual[i, :])
        #         acoustic.append(_acoustic[i, :])

        words = np.asarray(_words)
        visual = np.asarray(_visual)
        acoustic = np.asarray(_acoustic)

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

    # turn off the word2id - define a named function here to allow for pickling
    def return_unk():
        return UNK

    word2id.default_factory = return_unk

    # input dimension 
    # text_size = 300
    # visual_size = 47
    # acoustic_size = 74
    # should be in order of audio, video, text for TFN 
    # input_dims = [acoustic_size, visual_size, text_size]
    # hidden_sizes = [int(text_size * 1.5), int(visual_size * 1.5), int(acoustic_size * 1.5)]

    return train, valid, test
