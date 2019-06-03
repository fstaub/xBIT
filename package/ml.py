import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import package.screen as screen


# -------------------------------------
# the basic class for neural networks
# -------------------------------------
class NN():
    def __init__(self):
        self.input_dim = len(self.inputs['Observables'])

        self.train_lh = self.inputs['ML']['TrainLH']

        if self.train_lh:
            self.output_dim = 1
        else:
            self.output_dim = len(self.inputs['Observables'])

        self.epochs = self.inputs['ML']['Epochs']
        self.iterations = self.inputs['Setup']['Iterations']
        self.neurons = self.inputs['ML']['Neurons']
        self.LR = self.inputs['ML']['LR']
        self.distance_penalty = self.inputs['ML']['DensityPenality']
        self.use_classifier = self.inputs['ML']['Classifier']

    def set_predictor(self):
        ''' network which predicts numerical values for observables'''
        layers = [nn.Linear(self.input_dim, self.neurons[0]),
                  nn.ReLU(), nn.Dropout(0.1)]
        for i in range(1, len(self.neurons)):
            layers.append(nn.Linear(self.neurons[i - 1], self.neurons[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(self.neurons[-1], self.output_dim))
        self.predictor = nn.Sequential(*layers)

        self.predictor_optimizer = optim.Adam(self.predictor.parameters(),
                                              lr=self.LR)
        self.predictor_criterion = nn.MSELoss()

    def set_classifier(self):
        ''' network which checks if a point is valid or not'''
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, 50), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(50,50), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(50, 2), nn.Sigmoid())

        self.classifier_optimizer = optim.Adam(self.classifier.parameters(),
                                               lr=self.LR)
        self.classifier_criterion = nn.BCELoss()

    # training the neural network
    def train(self, name, x_val, y_val, log):
        log.info("Training the neural network")
        if self.config.cursesQ:
            screen.train_nn(self.config.screen, 0, 0)

        torch.set_num_threads(self.inputs['Setup']['Cores'])

        best = 1.0e6
        wait = 0
        batch_size = 128
        nr_batches = max(1, int(len(x_val) / batch_size))
        patience = max(100, self.epochs / 5 / nr_batches)

        if name is self.predictor:
            criterion = self.predictor_criterion
            optimizer = self.predictor_optimizer
        else:
            criterion = self.classifier_criterion
            optimizer = self.classifier_optimizer

        rand = list(zip(x_val, y_val))
        random.shuffle(rand)
        x_r, y_r = zip(*rand)

        x_train = torch.FloatTensor(x_r[:int(0.8 * len(x_r))])
        y_train = torch.FloatTensor(y_r[:int(0.8 * len(x_r))])

        x_test = torch.FloatTensor(x_r[int(0.8 * len(x_r)) + 1:])
        y_test = torch.FloatTensor(y_r[int(0.8 * len(x_r)) + 1:])

        # for mini batch
        # permutation = torch.randperm(len(x_train))
        permutation = torch.randperm(x_train.size()[0])

        for epoch in range(self.epochs):
            name.train()

            # mini batches
            for i in range(0, x_train.size()[0], batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = x_train[indices], y_train[indices]
                y = name(Variable(batch_x))
                self.loss = criterion(y, Variable(batch_y))

                self.loss.backward()
                optimizer.step()

            # calculate loss with test data
            name.eval()
            y_pred = name(Variable(x_test)).data.cpu()
            loss = criterion(Variable(y_pred), Variable(y_test))

            if epoch % 100 == 0:
                if self.config.cursesQ:
                    screen.train_nn(self.config.screen, epoch, loss)
                log.debug("Epoch: %i;  loss: %f" % (epoch, loss))

            # Simple implementation of early stopping
            if loss < best:
                best = loss
                wait = 0
                torch.save({'state_dict': name.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss': self.loss},
                           os.path.join(self.temp_dir, "checkpoint.pth.tar"))
            else:
                wait = wait + 1
            if wait > patience or epoch == (self.epochs - 1):
                log.info("Stopped after %i epochs" % epoch)
                checkpoint = torch.load(
                    os.path.join(self.temp_dir, "checkpoint.pth.tar")
                )
                name.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                self.loss = checkpoint['loss']
                break
        log.info("Training Data: %i points, loss: %f" % (len(x_val), best))
        if self.config.cursesQ:
            screen.status_nn(self.config.screen, len(x_val), best)

    def scale_data(self, input, log):
        '''Scale the observables: using a log for parameters which
           vary by several orders of magnitude'''
        self.scalings = []
        transposed = np.array(input).T
        data_out = None
        for data in transposed:
            if max(data) / min(data) > 100:  # large spread in results = log
                if data_out is None:
                    data_out = np.array([np.log(data)])
                else:
                    data_out = np.concatenate((data_out, [np.log(data)]),
                                              axis=0)
                self.scalings.append("log")
            else:
                if data_out is None:
                    data_out = np.array([(data)])
                else:
                    data_out = np.concatenate((data_out, [data]), axis=0)
                self.scalings.append("id")
        log.info("Using the following scalings for the observables: %s"
                 % str(self.scalings))
        return((data_out.T).tolist())

    def rescale(self, input):
        '''Rescale the predictions of the NN'''
        transposed = np.array(input).T
        data_out = None
        for data, scale in zip(transposed, self.scalings):
            if scale == "log":
                if data_out is None:  # why doesn't it work with empty lists?
                    data_out = np.array([np.exp(data)])
                else:
                    data_out = np.concatenate((data_out, [np.exp(data)]),
                                              axis=0)
            elif scale == "id":
                if data_out is None:
                    data_out = np.array([(data)])
                else:
                    data_out = np.concatenate((data_out, [data]), axis=0)
        return((data_out.T).tolist())
