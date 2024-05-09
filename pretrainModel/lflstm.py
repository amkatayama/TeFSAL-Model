import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from collections import defaultdict
from mosi_to_tensor import preprocess_mosi, multi_collate

import argparse

train, valid, test = preprocess_mosi()

# construct dataloaders, dev and test could use around ~X3 times batch size since no_grad is used during eval
batch_sz = 56
train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate)
valid_loader = DataLoader(valid, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)
test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)

# let's define a simple model that can deal with multimodal vriable length sequence
class LFLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, fc1_size, output_size, dropout_rate):
        super(LFLSTM, self).__init__()
        self.input_size = input_sizes
        self.hidden_size = hidden_sizes
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # defining modules - two layer bidirectional LSTM with layer norm in between
        # self.embed = nn.Embedding(len(word2id), input_sizes[0])
        self.trnn1 = nn.LSTM(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.trnn2 = nn.LSTM(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = nn.LSTM(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = nn.LSTM(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = nn.LSTM(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = nn.LSTM(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.fc1 = nn.Linear(sum(hidden_sizes)*4, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))
        self.bn = nn.BatchNorm1d(sum(hidden_sizes)*4)
        # self.bn = nn.BatchNorm1d(19668992*56/2)

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        sequence = sequence.float()  # Convert sequence to float
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

        
    # tensor fusion
    def fusion(self, sentences, visual, acoustic, lengths):
        batch_size = lengths.size(0)
        sentences = sentences.float()  # Convert sentences to float
        visual = visual.float()  # Convert visual to float
        acoustic = acoustic.float()  # Convert acoustic to float
        
        # extract features from text modality
        final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        
        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        
        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)

        # simple late fusion -- concatenation + normalization
        h = torch.cat((final_h1t, final_h2t, final_h1v, final_h2v, final_h1a, final_h2a),
                       dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        return h
    
    def forward(self, sentences, visual, acoustic, lengths):
        batch_size = lengths.size(0)
        h = self.fusion(sentences, visual, acoustic, lengths)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o

from tqdm import tqdm_notebook
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='model name')
    parser.add_argument('modelName', type=str, help='Model name')
    args = parser.parse_args()

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    CUDA = torch.cuda.is_available()
    MAX_EPOCH = 3

    text_size = 300
    # visual_size = 713
    visual_size = 430
    # visual_size = 47  # facet 41
    # visual_size = 35  # facet 42
    acoustic_size = 74

    # define some model settings and hyper-parameters
    input_sizes = [text_size, visual_size, acoustic_size]
    hidden_sizes = [int(text_size * 1.5), int(visual_size * 1.5), int(acoustic_size * 1.5)]
    fc1_size = sum(hidden_sizes) // 2
    dropout = 0.25
    output_size = 1
    curr_patience = patience = 8
    num_trials = 3
    grad_clip_value = 1.0
    weight_decay = 0.1

    # if os.path.exists(CACHE_PATH):
    #     pretrained_emb, word2id = torch.load(CACHE_PATH)
    # elif WORD_EMB_PATH is not None:
    #     pretrained_emb = load_emb(word2id, WORD_EMB_PATH)
    #     torch.save((pretrained_emb, word2id), CACHE_PATH)
    # else:
    #     pretrained_emb = None

    # model = LFLSTM(input_sizes, hidden_sizes, fc1_size, output_size, dropout)
    model = LFLSTM(input_sizes, hidden_sizes, fc1_size, output_size, dropout)
    # if pretrained_emb is not None:
    #     model.embed.weight.data = pretrained_emb
    # model.embed.requires_grad = False
    optimizer = Adam([param for param in model.parameters() if param.requires_grad], weight_decay=weight_decay)

    if CUDA:
        model.cuda()
    criterion = nn.L1Loss(reduction='sum')
    criterion_test = nn.L1Loss(reduction='sum')
    best_valid_loss = float('inf')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    lr_scheduler.step() # for some reason it seems the StepLR needs to be stepped once first
    train_losses = []
    valid_losses = []
    for e in range(MAX_EPOCH):
        model.train()
        train_iter = tqdm_notebook(train_loader)
        train_loss = 0.0
        for batch in train_iter:
            model.zero_grad()
            t, v, a, y, l = batch
            batch_size = t.size(0)
            if CUDA:
                t = t.cuda()
                v = v.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()
            y_tilde = model(t, v, a, l)
            loss = criterion(y_tilde, y)
            loss.backward()
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip_value)
            optimizer.step()
            train_iter.set_description(f"Epoch {e}/{MAX_EPOCH}, current batch loss: {round(loss.item()/batch_size, 4)}")
            train_loss += loss.item()
        train_loss = train_loss / len(train)
        train_losses.append(train_loss)
        print(f"Epoch {e} Training loss: {round(train_loss, 4)}")

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for batch in valid_loader:
                model.zero_grad()
                t, v, a, y, l = batch
                if CUDA:
                    t = t.cuda()
                    v = v.cuda()
                    a = a.cuda()
                    y = y.cuda()
                    l = l.cuda()
                y_tilde = model(t, v, a, l)
                loss = criterion(y_tilde, y)
                valid_loss += loss.item()
        
        valid_loss = valid_loss/len(valid)
        valid_losses.append(valid_loss)
        print(f"Epoch {e} Validation loss: {round(valid_loss, 4)}")
        print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("Found new best model on dev set!")
            torch.save(model.state_dict(), f'{args.modelName}.std')
            torch.save(optimizer.state_dict(), f'optim_{args.modelName}.std')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= -1:
                print("Running out of patience, loading previous best model.")
                num_trials -= 1
                curr_patience = patience
                model.load_state_dict(torch.load(f'{args.modelName}.std'))
                optimizer.load_state_dict(torch.load(f'optim_{args.modelName}.std'))
                lr_scheduler.step()
                print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        
        if num_trials <= 0:
            print("Running out of patience, early stopping.")
            break

    model.load_state_dict(torch.load(f'{args.modelName}.std'))
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            model.zero_grad()
            t, v, a, y, l = batch
            if CUDA:
                t = t.cuda()
                v = v.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()
            y_tilde = model(t, v, a, l)
            loss = criterion_test(y_tilde, y)
            y_true.append(y_tilde.detach().cpu().numpy())
            y_pred.append(y.detach().cpu().numpy())
            test_loss += loss.item()
    print(f"Test set performance: {test_loss/len(test)}")
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    y_true_bin = y_true >= 0
    y_pred_bin = y_pred >= 0
    bin_acc = accuracy_score(y_true_bin, y_pred_bin)
    print("True values: ", y_true[:20], "Predicted values: ", y_pred[:20])
    print(f"Test set accuracy is {bin_acc}")