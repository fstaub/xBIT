from package.scanning import Scan as Scan
from package.ml import NN as NN


import itertools
import numpy as np
import math

# from pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# auxiliary functions already included in xBIT
from package.aux import LINEAR_DIFF as LINEAR_DIFF
from package.aux import exp_safe as exp_safe
from package.aux import limit_lh as limit_lh


scan_name = "MLS"

class NewScan(Scan, NN):
    """Scanner class for Machine Learning Scans

       Algorithm to improve parameter space sampling using a neural network
       Basic idea:
         1) Start with random sample
         2) Train NN
         3) Use NN to propose new points with a 'good likelihood'
        Approach proposed by J. Ren,  L. Wu,  J.M. Yang,  J. Zhao
    """

    def __init__(self, inputs, config):
        Scan.__init__(self, inputs, config)
        NN.__init__(self)

    def generate_parameter_points(self, vars, points):
        # generate random points in the given range
        all = []
        for _ in range(points):
            all += [[eval(vars[xx]) for xx in vars]]
        return all

    def run(self):
        self.config.log.info("\n\n##### Iteration: %i #####" % (1))
        if self.config.cursesQ:
            self.screen.iteration_nn(self.config.screen, 1, self.iterations)

        # first run: just a ordinary random scan to get the initial sample
        save_points = self.inputs['Setup']['Points']
        self.inputs['Setup']['Points'] = 5 * self.inputs['Setup']['Points']
        self.runner.run(self.config.log, sample=[])
        self.inputs['Setup']['Points'] = save_points

        while not self.input_and_observables.empty():
            self.all_data.append(self.input_and_observables.get())
        while not self.valid_points.empty():
            self.all_valid.append(self.valid_points.get())
        while not self.invalid_points.empty():
            self.all_invalid.append(self.invalid_points.get())

        # use neural networks for the next iterations
        for i in range(1, self.iterations):
            x = [xx[0] for xx in self.all_data]
            y = [xx[1] for xx in self.all_data]
            if self.train_lh:
                y_scale = [[math.log(self.likelihood(yy) + 1.0e-10)]
                           for yy in y]
            else:
                y_scale = self.scale_data(y, self.config.log)

            self.set_predictor()
            self.train(self.predictor, x, y_scale, self.config.log)

            # Include a classifier to distinguish valid/invalid points
            if self.use_classifier:
                self.set_classifier()
                x_class = self.all_valid + self.all_invalid[0:min([len(self.all_valid),len(self.all_invalid)-1])]
                y_class = [[0.99, 0.01] for j in self.all_valid] + \
                          [[0.01, 0.99] for j in self.all_invalid[0:min([len(self.all_valid),len(self.all_invalid)-1])]]
                self.train(self.classifier, x_class, y_class, self.config.log)

            proposed_points = self.new_points(
                self.inputs['Setup']['Points'],
                len(self.inputs['Variables'])**2 * self.inputs['Setup']['Points']                
            )
            self.config.log.info("##### Iteration: %i #####" % (i + 2))
            if self.config.cursesQ:
                self.screen.iteration_nn(self.config.screen, i + 1, self.iterations)
            self.runner.run(self.config.log, sample=proposed_points)
            while not self.input_and_observables.empty():
                self.all_data.append(self.input_and_observables.get())

    def guess_LH(self, x_in):
        # Run predictor
        if self.train_lh:
            y_try = self.predictor(
                Variable(torch.FloatTensor(x_in))).data.cpu().numpy()
        else:
            y_try = self.rescale(self.predictor(
                Variable(torch.FloatTensor(x_in))).data.cpu().numpy())

        # Run classifier, if included
        if self.use_classifier:
            y_class = self.classifier(
                Variable(torch.FloatTensor(x_in))).data.cpu().numpy()
        else:
            y_class = [[0.99, 0.01] for _ in y_try]

        # Filter valid points and calculate likelihood
        lh_valid = []
        x_valid = []
        for y, x, yc in zip(y_try, x_in, y_class):
            if (yc[0] > yc[1]):
                if self.train_lh:
                    lh_valid.append(exp_safe(y))
                else:
                    lh_valid.append(self.likelihood(y))
                x_valid.append(x)

        # Calculate final likelihood (optionally with distance penalty)
        if not self.distance_penalty:
            likelihood = lh_valid
        else:
            likelihood = self.min_Delta_LH(lh_valid, x_valid)

        return likelihood, x_valid

    def new_points(self, good_points, points):
        # estimate new points with good likelihood
        self.predictor.eval()
        best_lh=[]
        increase=1
        while (len(best_lh)<5):
            x_try = self.generate_parameter_points(self.inputs['Variables'], max(increase*points,100000))
            likelihood, x_try = self.guess_LH(x_try)
            best_lh = [l for l in likelihood if l < 1]
            self.config.log.debug("Proposed LHs: %s" % (str(best_lh)))
            increase += 1
            if increase > 100:
                break
        if len(best_lh) > 0:
            best_lh = max(best_lh)
            count = 1
            count_all = 0
            self.config.log.info("Best LH proposed by NN: %s" % (str(best_lh)))
            while count < int(0.9 * good_points):
                count_all = count_all + 1            
                for l, x in zip(likelihood, x_try):
                    if 2. > l > limit_lh(count_all) * best_lh:  # 2 > might be necessary for the LH fit because some good points are predicted to have slighlyt higher LH;
                        self.config.log.debug("Accepted point %i by NN: LH: %s"
                                    % (count, str(l)))
                        if count == 1:
                            new_good_points = [x]
                        else:
                            new_good_points = np.concatenate((new_good_points, [x]))
                        count = count + 1
                        if count > int(0.9 * good_points):
                            break
                self.config.log.debug("Accepted Points by NN: %i after %i iterations" % (count, count_all))            
                x_try = self.generate_parameter_points(self.inputs['Variables'], max(increase*points,100000))            
                likelihood, x_try = self.guess_LH(x_try)
                # print("x_try", likelihood)

            # add 10% random points
            new_x = np.concatenate((
                self.generate_parameter_points(self.inputs['Variables'], int(0.1 * good_points)),
                new_good_points
            ))
        else:
            # no valid points found by the NN! That's likely an overconstraint from the classifier. 
            # in order to contue: let's take 100% random points
            self.config.log.info("No valid points predicted by the NN!")
            new_x = self.generate_parameter_points(self.inputs['Variables'], good_points)
        return new_x

    def min_Delta_LH(self, lh, xval):
        '''Function to calculate the minimal 'distance' of a new point
        to previously sampled points'''
        all_points = np.array([x[0] for x in self.all_data])
        norm = np.array([eval(x.replace("np.uniform", "LINEAR_DIFF"))
                         for x in self.inputs['Variables'].values()])
        lh_out = []
        for l, x in zip(lh, xval):
            min_d = np.amax(np.abs((all_points - x) / norm),
                            axis=1).min()
            lh_out.append(l * min_d**2)
        return lh_out     
