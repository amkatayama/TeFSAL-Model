import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from collections import defaultdict
from mosi_to_tensor import preprocess_mosi, multi_collate
# from tensor_fusion import TensorFusionNetwork  

train, valid, test = preprocess_mosi()

batch_sz = 56
train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate)
valid_loader = DataLoader(valid, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)
test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)

class TensorFusionNetwork(nn.Module):
    def __init__(self, text_dim, visual_dim, acoustic_dim):
        super(TensorFusionNetwork, self).__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.acoustic_dim = acoustic_dim

    def forward(self, text, visual, acoustic):
        # Text, Visual, and Acoustic inputs are of shape [seq_len, batch_size, feature_dim]

        # Add a dimension for the bias term in the tensor (concatenating ones)
        # Add ones across the seq_len and batch_size dimensions
        ones = torch.ones(text.size(0), text.size(1), 1, device=text.device)

        text = torch.cat([text, ones], dim=2)  # New shape [seq_len, batch_size, text_dim+1]
        visual = torch.cat([visual, ones], dim=2)  # New shape [seq_len, batch_size, visual_dim+1]
        acoustic = torch.cat([acoustic, ones], dim=2)  # New shape [seq_len, batch_size, acoustic_dim+1]

        # Compute the outer product across all dimensions for each time step and batch element
        # Resulting shape will be [seq_len, batch_size, text_dim+1, visual_dim+1, acoustic_dim+1]
        fusion = torch.einsum('sbt,sbv,sba->sbtva', text, visual, acoustic)

        # Flatten the fusion tensor to make it suitable for processing by subsequent layers
        # Flatten all dimensions except the sequence and batch dimensions
        fusion = fusion.flatten(start_dim=2)

        return fusion

class TFLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, fc1_size, output_size, dropout_rate):
        super(TFLSTM, self).__init__()
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # Instantiate the Tensor Fusion Network with adjusted dimensions
        self.tensor_fusion_network = TensorFusionNetwork(input_sizes[0] + 1, input_sizes[1] + 1, input_sizes[2] + 1)
        
        # Calculate the size of the flattened fusion tensor
        fusion_dim = (input_sizes[0] + 1) * (input_sizes[1] + 1) * (input_sizes[2] + 1)
        
        # One LSTM to process the fused input
        self.rnn = nn.LSTM(fusion_dim, sum(hidden_sizes) // 3, bidirectional=True, num_layers=2)  # Adjust hidden_size appropriately

        self.fc1 = nn.Linear(sum(hidden_sizes) // 3 * 2, fc1_size)  # *2 for bidirectional
        self.fc2 = nn.Linear(fc1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm((sum(hidden_sizes) // 3 * 2,))

    def forward(self, sentences, visual, acoustic, lengths):
        # Assuming fusion gives you a batch-first format: [batch_size, feature_size]
        fused_input = self.tensor_fusion_network(sentences, visual, acoustic)
        
        # Reshape to add a sequence length of 1, if your model treats each sample independently
        fused_input = fused_input.unsqueeze(1)  # Now shape [batch_size, 1, feature_size]

        # Process sequence through RNN
        packed_input = pack_padded_sequence(fused_input, lengths, enforce_sorted=False)
        packed_output, (hidden, _) = self.rnn(packed_input)
        padded_output, _ = pad_packed_sequence(packed_output)

        normed_output = self.layer_norm(padded_output)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        h = self.fc1(final_hidden)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)

        return o

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='model name')
    parser.add_argument('modelName', type=str, help='Model name')
    args = parser.parse_args()

    from tqdm import tqdm_notebook
    from torch.optim import Adam, SGD
    from sklearn.metrics import accuracy_score

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    CUDA = torch.cuda.is_available()
    MAX_EPOCH = 100

    text_size = 300
    visual_size = 430
    acoustic_size = 74

    # define some model settings and hyper-parameters
    input_sizes = [text_size, visual_size, acoustic_size]
    hidden_sizes = [int(text_size * 1.5), int(visual_size * 1.5), int(acoustic_size * 1.5)]
    confounder_size = 50
    fc1_size = sum(hidden_sizes) // 2
    dropout = 0.25
    output_size = 1
    curr_patience = patience = 8
    num_trials = 3
    grad_clip_value = 1.0
    weight_decay = 0.1

    model = TFLSTM(input_sizes, hidden_sizes, fc1_size, output_size, dropout)

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
            torch.save(model.state_dict(), f'./modelparams/{args.modelName}.std')
            torch.save(optimizer.state_dict(), f'./modelparams/optim_{args.modelName}.std')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= -1:
                print("Running out of patience, loading previous best model.")
                num_trials -= 1
                curr_patience = patience
                model.load_state_dict(torch.load(f'./modelparams/{args.modelName}.std'))
                optimizer.load_state_dict(torch.load(f'./modelparams/optim_{args.modelName}.std'))
                lr_scheduler.step()
                print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        
        if num_trials <= 0:
            print("Running out of patience, early stopping.")
            break

    model.load_state_dict(torch.load(f'./modelparams/{args.modelName}.std'))
    print("Successfully loaded model!")
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
    # Assuming y_true and y_pred are your arrays of true and predicted labels
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # Assuming y_true and y_pred need to be flattened or adjusted to match dimensions
    y_true = y_true.flatten()
    y_pred = y_pred[:, 0]  # Assuming we need the first column only if it's the actual predictions

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)