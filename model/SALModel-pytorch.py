import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

epsilon = 1e-7
seed = 0

# Define activation functions
def tanh(x):
    return torch.tanh(x)

def relu(x):
    return F.relu(x)

def linear(x):
    return x

def sigmoid(x):
    return torch.sigmoid(x)

activation = tanh
activation_cnn = tanh
activation_output = F.softmax

class SALModel(nn.Module):
    def __init__(self, continuous, n_class,
                 filterShapeText, poolSizeText, imageShapeText, dropOutSizeText,
                 x_train_fix, y_train, groupId,  x_valid_fix, y_valid, groupId_valid, x_test_fix, y_test,
                 b1=5, b2=1, batch_size=100, learning_rate=0.001, lam=0, dropOutRate1=0.5, dropOutRate2=0.5,
                 sigmaBias=0.01, yLam=0.1):
        super(SALModel, self).__init__()
        self.continuous = continuous
        self.N, self.features = x_train_fix.shape
        self.groupIDsize = groupId.shape[1]
        self.validSize = y_train.shape[0]
        self.testSize = x_test_fix.shape[0]
        self.b1 = np.float32(b1)
        self.b2 = np.float32(b2)
        self.learning_rate = np.float32(learning_rate)
        self.lam = np.float32(lam)
        self.dropOutRate1 = np.float32(dropOutRate1)
        self.dropOutRate2 = np.float32(dropOutRate2)
        self.sigmaBias = np.float32(sigmaBias)
        self.ylam = np.float32(yLam)
        self.batch_size = batch_size
        self.filterShapeText = filterShapeText
        self.poolSizeText = poolSizeText
        self.dropOutSizeText = dropOutSizeText
        self.imageShapeText = imageShapeText
        sigma_init = 1
        create_weight = lambda dim_input, dim_output: torch.tensor(np.random.normal(0, sigma_init, (dim_input, dim_output)).astype(np.float32))
        create_bias = lambda dim_output: torch.zeros(dim_output)
        create_weight_zeros = lambda dim_input, dim_output: torch.zeros(dim_input, dim_output)
        
        # Fix Effect Convolution Layer 1
        fan_in = np.prod(filterShapeText[0][1:])
        fan_out = (filterShapeText[0][0] * np.prod(filterShapeText[0][2:]) // np.prod(poolSizeText[0]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W_fh0 = nn.Parameter(torch.tensor(np.random.uniform(low=-W_bound, high=W_bound, size=filterShapeText[0]), dtype=torch.float32))
        self.b_fh0 = nn.Parameter(create_bias((filterShapeText[0][0],)))
        
        # Fix Effect Convolution Layer 2
        fan_in = np.prod(filterShapeText[1][1:])
        fan_out = (filterShapeText[1][0] * np.prod(filterShapeText[1][2:]) // np.prod(poolSizeText[1]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W_fh1 = nn.Parameter(torch.tensor(np.random.uniform(low=-W_bound, high=W_bound, size=filterShapeText[1]), dtype=torch.float32))
        self.b_fh1 = nn.Parameter(create_bias((filterShapeText[1][0],)))
        
        self.W_rh = nn.Parameter(create_weight(self.groupIDsize, dropOutSizeText[0][0]))
        self.b_rh = nn.Parameter(create_bias(dropOutSizeText[0][0]))
        
        self.W_ch = nn.Parameter(create_weight(dropOutSizeText[0][0], dropOutSizeText[0][1]))
        self.b_ch = nn.Parameter(create_bias(dropOutSizeText[0][1]))
        
        self.W_yh = nn.Parameter(create_weight(dropOutSizeText[0][1], n_class))
        self.b_yh = nn.Parameter(create_bias(n_class))
        
        self.params = nn.ParameterDict({"W_fh0": self.W_fh0, "b_fh0": self.b_fh0, "W_fh1": self.W_fh1, "b_fh1": self.b_fh1,
                                        "W_rh": self.W_rh, "b_rh": self.b_rh, "W_ch": self.W_ch, "b_ch": self.b_ch,
                                        "W_yh": self.W_yh, "b_yh": self.b_yh})
        
        self.params_fixed = nn.ParameterDict({"W_ch": self.W_ch, "b_ch": self.b_ch, "W_yh": self.W_yh, "b_yh": self.b_yh})
        
        self.params_random = nn.ParameterDict({"W_rh": self.W_rh, "b_rh": self.b_rh})
        
        # Define other necessary parameters and layers
        
    def encoder(self, x, b):
        log_sigma = torch.matmul(b, self.W_rh) + self.b_rh
        
        return x, log_sigma
    
    def encoder_rand(self, x, b):
        x, log_sigma = self.encoder(x, b)
        if self.dropOutRate2 > 0:
            mask = torch.bernoulli(torch.ones_like(log_sigma) * (1 - self.dropOutRate2))
            log_sigma = log_sigma * mask
        return x, log_sigma
    
    def textCNN(self, t):
        layer0_input = t.view(self.imageShapeText[0])
        conv_out0 = F.conv2d(layer0_input, self.W_fh0, bias=self.b_fh0)
        pooled_out0 = F.max_pool2d(conv_out0, self.poolSizeText[0])
        output0 = activation_cnn(pooled_out0 + self.b_fh0.view(1, -1, 1, 1))
        conv_out1 = F.conv2d(output0, self.W_fh1, bias=self.b_fh1)
        pooled_out1 = F.max_pool2d(conv_out1, self.poolSizeText[1])
        output1 = activation_cnn(pooled_out1 + self.b_fh1.view(1, -1, 1, 1))
        return output1.view(output1.size(0), -1)
    
    def sampler(self, mu, log_sigma):
        eps = torch.randn_like(mu)
        z = mu + (torch.exp(0.5 * log_sigma) - 1) * eps * 5e-1
        return z
    
    def sampler_rand(self, mu, log_sigma):
        return mu + (torch.exp(0.5 * log_sigma) - 1)
    
    def dropOutOutput(self, n):
        output2 = activation(F.linear(n, self.W_ch, self.b_ch))
        if self.dropOutRate1 > 0:
            mask = torch.bernoulli(torch.ones_like(output2) * (1 - self.dropOutRate1))
            output2 = output2 * mask
        return output2
    
    def dropOutOutput_rand(self, n):
        return activation(F.linear(n, self.W_ch, self.b_ch))
    
    def logisticOutput(self, o, y):
        p_y_given_x = activation_output(F.linear(o, self.W_yh, self.b_yh))
        y_pred = torch.argmax(p_y_given_x, dim=1)
        nll = -torch.mean(torch.log(p_y_given_x[torch.arange(y.shape[0]), y]))
        error = self.errors(y_pred, y)
        return y_pred, nll, error
    
    def decoder(self, x, z):
        h_decoder = relu(F.linear(z, self.W_zh, self.b_zh))
        
        if self.continuous:
            reconstructed_x = F.linear(h_decoder, self.W_hxmu, self.b_hxmu)
            log_sigma_decoder = F.linear(h_decoder, self.W_hxsigma, self.b_hxsigma)
            logpxz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) - 
                      0.5 * ((x - reconstructed_x) ** 2 / torch.exp(log_sigma_decoder))).sum(dim=1, keepdim=True)
        else:
            reconstructed_x = torch.sigmoid(F.linear(h_decoder, self.W_hx, self.b_hx))
            logpxz = -F.binary_cross_entropy(reconstructed_x, x, reduction='sum').unsqueeze(1)
        
        return reconstructed_x, logpxz
    
    def errors(self, y_pred, y):
        if y.dim() != y_pred.dim():
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return torch.mean((y_pred != y).float())
        else:
            raise NotImplementedError()