__author__ = 'Haohan Wang'

import sys
sys.path.append('../')

import numpy as np
import theano

from utility.dataLoader import load_data_self_predict

from .SALModel import SALModel
from theano import config


import time
from sklearn.metrics import accuracy_score, f1_score

if __name__ == '__main__':
    # never change parameters
    np.random.seed(0)
    continuous = False
    print(config.device)
    print(config.floatX)

    # training configuration
    n_epochs = 1
    batch_size = 50
    learning_rate = 1e-3
    lamda = 0
    dropout1 = 0
    dropout2 = 0
    ylam = 1e-3

    imageShapeText = ((batch_size, 1, 60, 300),(batch_size, 25, 29, 149))
    filterShapeText = ((25,1,3,3),(50, 25, 2, 2))
    poolSizeText = ((2,2),(2,2))
    dropOutSizeText = ((50 * 14 * 74, 200),)

    # Loading Features
    datasets = load_data_self_predict('text')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    groupId1 = datasets[3]
    groupId2 = datasets[4] # this is actually useless

    # groupId = theano.shared(groupId.astype(theano.config.floatX), name="groupId")

    cvCount = 0

    rs = []
    fs = []

    model = SALModel(continuous, 2, filterShapeText, poolSizeText, imageShapeText, dropOutSizeText,
                     train_set_x, train_set_y,  groupId1, valid_set_x, valid_set_y, groupId2, test_set_x, test_set_y,
                     b1 = 0.1, b2=0.01,
             batch_size=batch_size, learning_rate=learning_rate, lam=lamda, dropOutRate1=dropout1, dropOutRate2=dropout2, yLam=ylam)

    model.load_parameters('../params/textModelParams.npy')

    cvCount += 1
    batch_order = np.arange(int(model.N / model.batch_size))
    test_batch_order = np.arange(int(valid_set_y.shape[0] / model.batch_size))
    test_batch_order2 = np.arange(int(test_set_y.shape[0] / model.batch_size))
    epoch = 0
    # print 'set ', count
    bestError = np.inf
    lcount = []
    best_params = model.params
    bestTestError = 0
    rw = None
    rb = None
    while epoch < n_epochs:
        epoch += 1
        start = time.time()
        train_error = 0.

        miniepoch = 0
        print('random effects only tuning:', end=' ')
        mini_r_m = []
        besterror_random = np.inf
        while miniepoch < 50:
            mini_r = []
            for batch in batch_order:
                error_mini_r = model.update_random(batch)
                mini_r.append(error_mini_r)
            merr = np.mean(mini_r)
            mini_r_m.append(merr)
            miniepoch += 1

            if merr < besterror_random:
                besterror_random = merr
                rw = model.params['W_rh'].get_value(True)
                rb = model.params['b_rh'].get_value(True)
        model.update_random_params(rw, rb)

        print(np.mean(mini_r_m), besterror_random)
        print('')

        print('fixed effects only tuning:', end=' ')
        mini_f_m = []
        miniepoch = 0
        while miniepoch < 50:
            mini_f = []
            for batch in batch_order:
                error_mini_f = model.update_fixed(batch)
                mini_f.append(error_mini_f)
            merf = np.mean(mini_f)
            train_error = merf
            mini_f_m.append(merf)
            miniepoch += 1

            print("Epoch {3} MINI Epoch {0} finished. Cost: {1}, time: {2}".format(miniepoch, train_error, time.time() - start, epoch), end=' ')
            batch_error = 0
            for mbatch in test_batch_order:
                batch_error += model.test(mbatch)

            validerror = batch_error/len(test_batch_order)
            print('Validation error', validerror, end=' ')
            if validerror not in lcount:
                lcount.append(validerror)
            print('error combination', len(lcount))

            if validerror < bestError:
                bestError = validerror
                best_params = model.get_parameters()
                np.save('../params/textModelParams', best_params)
                print('NEW Best!', end=' ')
                error = []
                for mbatch2 in test_batch_order2:
                    error.append(model.predict(mbatch2))
                bestTestError = np.mean(error)
                print('test error', bestTestError)

    print('best test model', bestTestError, 'with validation error', bestError)
