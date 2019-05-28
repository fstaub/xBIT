import json
import os
import shutil
import time
# import datetime
import multiprocessing as mp
import math
# import collections
import itertools
import numpy as np
# import curses
from six.moves import queue

from aux import LINEAR as LINEAR
from aux import LINEAR_DIFF as LINEAR_DIFF
from aux import LOG as LOG
from aux import gauss as gauss
from aux import logL as logL
from aux import exp_safe as exp_safe
from aux import limit_lh as limit_lh

# for the NN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# home-brewed
import running
import xslha
import ml
# import debug
import screen

# from aux import dd

# ----------------------------------------------------------
# Scan Class
# All information necessary to 'define' a scan
# ----------------------------------------------------------

class Scan:
    """Main scanner class"""

    def __init__(self, screen, input, main_dir, temp_dir, debug, curses, log):
        log.info('Initialise scan: %s' % input['Setup']['Name'])

        # main information about the scan
        self.setup = input['Setup']
        self.codes = input['Included_Codes']
        self.blocks = input['Blocks']
        self.observables = input['Observables']
        self.variables = input['Variables']
        self.ml = input['ML']

        self.Short = input['Short']


        self.screen = screen

        # directories and log file
        self.main_dir = main_dir
        self.temp_dir = temp_dir
        self.log = log
        self.curses = curses
        self.debug = debug

        self.distance_penalty = True

        self.scalings = ["id"] * len(self.observables)

        # for input parameter and values of observables
        self.all_data = []

        # String to Bool
        for c in self.codes:
            self.codes[c] = eval(self.codes[c])

        # load the settings-file
        self.parse_settings(self.setup['Settings'], log)
        self.set_up_codes()
        self.make_out_dir()
        self.output_file = os.path.join(self.main_dir, "Output",
                                        self.setup['Name'], "SpectrumFiles")

        if input['Setup']['Cores'] > 1:
            # initialise queues needed for multicore runs
            self.input_and_observables = mp.Queue()
            self.all_points = mp.Queue()
            self.valid_points = mp.Queue()
            self.invalid_points = mp.Queue()
        else:
            self.input_and_observables = queue.Queue()
            self.all_points = queue.Queue()
            self.valid_points = queue.Queue()
            self.invalid_points = queue.Queue()

        # for training the classifier
        self.all_valid = []
        self.all_invalid = []

        # Initialse Runner class
        self.runner = running.Runner(self, log)

    def parse_settings(self, file, log):
        """ Parse the settings file which contains
            the paths,  executables,  etc. """
        log.info('Parse Settings file: %s' % file)
        with open("Settings/" + file) as json_data:
            d = json.load(json_data)
        self.settings = d
        log.debug('Settings: %s' % str(d))

    def set_up_codes(self):
        """ Set up the HEP Tools needed in the scan """
        self.spheno = running.HepTool("SPheno", self.settings['SPheno'],
                                      running.RunSPheno, self.log)
        if self.codes['HiggsBounds']:
            self.hb = running.HepTool("HiggsBounds",
                                      self.settings['HiggsBounds'],
                                      running.RunHiggs, self.log)
        if self.codes['HiggsSignals']:
            self.hs = running.HepTool("HiggsSignals",
                                      self.settings['HiggsSignals'],
                                      running.RunHiggs, self.log)
        if self.codes['MicrOmegas']:
            self.mo = running.HepTool("MicrOmegas",
                                      self.settings['MicrOmegas'],
                                      running.RunMicrOmegas, self.log)
        if self.codes['Vevacious']:
            self.vevacious = running.HepTool("Vevacious",
                                      self.settings['Vevacious'],
                                      running.RunVevacious, self.log)                                      

    def likelihood(self, x):
        """ calculate likelihood """
        lh = 1.
        for i, obs in enumerate(self.observables.values()):
            if self.scalings[i] == "id":
                lh = lh * gauss(x[i], obs["MEAN"], obs["VARIANCE"])
            elif self.scalings[i] == "log":
                lh = lh * logL(x[i], obs["MEAN"], obs["VARIANCE"])
        return lh

    def make_out_dir(self):
        """ Create output directory to store scan results"""
        self.out_dir = os.path.join(self.main_dir, "Output",
                                    self.setup['Name'])
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)

    def write_lh_file(self, point, dir, name):
        """ write Les Houches input file for given parameter point """
        lh = open(os.path.join(dir, name), "w+")
        for current_block in self.blocks:
            xslha.write_les_houches(current_block,
                                    self.blocks[current_block], point, lh)
        lh.close

    def start_run(self, log):
        self.start_time = time.time()
        log.info('Running scan %s' % str(self.setup['Name']))
        if self.curses:
            screen.show_setup(self.screen, self.setup, self.codes)

    def finish_run(self, log):
        log.info('Scan %s finished' % str(self.setup['Name']))
        print("All done!")
        print("Time Needed:               "
              + str(time.time() - self.start_time) + "s")


class Grid_Scan(Scan):
    """Scanner class for Grid Scans"""

    def __init__(self, screen, input, main_dir, temp_dir, debug, curses, log):
        Scan.__init__(self, screen, input, main_dir, temp_dir,
                      debug, curses, log)

    def run(self, log):
        self.start_run(log)
        self.runner.run(log, sample=[])
        self.finish_run(log)

    def generate_parameters(self, vars, points):
        all = []
        for x in vars:
            all.append(eval(vars[x]))
        temp = list(itertools.product(*all))
        all = [list(xx) for xx in temp]
        self.setup['Points'] = len(all)
        return all


class Random_Scan(Scan):
    """Scanner class for Random Scans"""

    def __init__(self, screen, input, main_dir, temp_dir, debug, curses, log):
        Scan.__init__(self, screen, input, main_dir, temp_dir,
                      debug, curses, log)

    def run(self, log):
            self.start_run(log)
            self.runner.run(log, sample=[])
            self.finish_run(log)

    def generate_parameters(self, vars, points):
        all = []
        for _ in range(points):
            all += [[eval(vars[xx]) for xx in vars]]
        return all


class MLS_Scan(Scan, ml.NN):
    """Scanner class for Machine Learning Scans

       Algorithm to improve parameter space sampling using a neural network
       Basic idea:
         1) Start with random sample
         2) Train NN
         3) Use NN to propose new points with a 'good likelihood'
        Approach proposed by from J. Ren,  L. Wu,  J.M. Yang,  J. Zhao
    """

# to-do/open questions:
#  - how to find islands?
#  - can one avoid a 'oversampling' of regions by defining a density-penality

    def __init__(self, screen, input, main_dir, temp_dir, debug, curses, log):
        Scan.__init__(self, screen, input, main_dir, temp_dir,
                      debug, curses, log)
        ml.NN.__init__(self)

    def generate_parameters(self, vars, points):
        all = []
        for _ in range(points):
            all += [[eval(vars[xx]) for xx in vars]]
        return all

    def run(self, log):
        log.info("\n\n##### Iteration: %i #####" % (1))
        if self.curses:
            screen.iteration_nn(self.screen, 1, self.iterations)
        # first run: just a ordinary random scan

        self.start_run(log)
        save_points = self.setup['Points']
        self.setup['Points'] = 5 * self.setup['Points']
        self.runner.run(log, sample=[])
        self.setup['Points'] = save_points

        while not self.input_and_observables.empty():
            self.all_data.append(self.input_and_observables.get())
        while not self.valid_points.empty():
            self.all_valid.append(self.valid_points.get())
        while not self.invalid_points.empty():
            self.all_invalid.append(self.invalid_points.get())

        # iterate
        for i in range(1, self.iterations):
            x = [xx[0] for xx in self.all_data]
            y = [xx[1] for xx in self.all_data]
            if self.train_lh:
                y_scale = [[math.log(self.likelihood(yy) + 1.0e-10)]
                           for yy in y]
            else:
                y_scale = self.scale_data(y, log)

            self.set_predictor()
            self.train(self.predictor, x, y_scale, log)

            if self.use_classifier:
                self.set_classifier()
                x_class = self.all_valid + self.all_invalid[0:min([len(self.all_valid),len(self.all_invalid)-1])]
#                print("valid, invalid:", len(self.all_valid), len(self.all_invalid))
                y_class = [[0.99, 0.01] for j in self.all_valid] + \
                          [[0.01, 0.99] for j in self.all_invalid[0:min([len(self.all_valid),len(self.all_invalid)-1])]]
                self.train(self.classifier, x_class, y_class, log)

            proposed_points = self.new_points(
                self.setup['Points'],
                #len(self.variables)**2 * (i + 1) * self.setup['Points']
                len(self.variables)**2 * self.setup['Points']                
            )
            log.info("##### Iteration: %i #####" % (i + 2))
            if self.curses:
                screen.iteration_nn(self.screen, i + 1, self.iterations)
            self.runner.run(log, sample=proposed_points)
            while not self.input_and_observables.empty():
                self.all_data.append(self.input_and_observables.get())

        self.finish_run(log)

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
#            print("classifier result: ", y_class)
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
            x_try = self.generate_parameters(self.variables, max(increase*points,100000))
            likelihood, x_try = self.guess_LH(x_try)
            best_lh = [l for l in likelihood if l < 1]
            self.log.debug("Proposed LHs: %s" % (str(best_lh)))
            increase += 1
            if increase > 100:
                break
        best_lh = max(best_lh)
        count = 1
        count_all = 0
        self.log.info("Best LH proposed by NN: %s" % (str(best_lh)))
        while count < int(0.9 * good_points):
            count_all = count_all + 1            
            for l, x in zip(likelihood, x_try):
                if 2. > l > limit_lh(count_all) * best_lh:  # 2 > might be necessary for the LH fit because some good points are predicted to have slighlyt higher LH;
                    self.log.debug("Accepted point %i by NN: LH: %s"
                                   % (count, str(l)))
                    if count == 1:
                        new_good_points = [x]
                    else:
                        new_good_points = np.concatenate((new_good_points, [x]))
                    count = count + 1
                    if count > int(0.9 * good_points):
                        break
            self.log.debug("Accepted Points by NN: %i after %i iterations" % (count, count_all))            
            x_try = self.generate_parameters(self.variables, max(increase*points,100000))            
            likelihood, x_try = self.guess_LH(x_try)
            # print("x_try", likelihood)

        # add 10% random points
        new_x = np.concatenate((
            self.generate_parameters(self.variables, int(0.1 * good_points)),
            new_good_points
        ))
        return new_x

#     def min_Delta(self, point, data):
#         '''Function to calculate the minimal 'distance' of a new point
#         to previously sampled points'''
#         sorted_data = sorted(data, key=lambda x: x[1])
#         min = 1.0e6
#         if dd(point, sorted_data[0]) > dd(point, sorted_data[-1]):
#             sorted_data.reverse()
#         for s in sorted_data:
#             new_dd = dd(point, s)
#             if new_dd < min:
#                 min = new_dd
# #            elif abs((point[0]-s[0])/(point[0]+s[0]))>min:
# #                break
#         return min

    def min_Delta_LH(self, lh, xval):
        '''Function to calculate the minimal 'distance' of a new point
        to previously sampled points'''
        all_points = np.array([x[0] for x in self.all_data])
        norm = np.array([eval(x.replace("LINEAR", "LINEAR_DIFF"))
                         for x in self.variables.values()])
        lh_out = []
        for l, x in zip(lh, xval):
            min_d = np.amax(np.abs((all_points - x) / norm),
                            axis=1).min()
            lh_out.append(l * min_d**2)
        return lh_out


class MCMC_Scan(Scan):
    """Scanner class for Basic MCMC Scans:
       A Marcov-Chain-Monte-Carlo class (MCMC) based
       on the Metropolis algorithm
    """

# to-do/questions:
#  - define convergence criterium
#  - check that tested points are not outside the scan ranges?
#  - let several chains run in parallel ( = > how to define number of chains in input?)
#  - gauss distribution always assumed  = > sufficient?

    def __init__(self, screen, input, main_dir, temp_dir, debug, curses, log):
        Scan.__init__(self, screen, input, main_dir, temp_dir,
                      debug, curses, log)
        self.last_step = []
        self.likelihood_last_step = 0
        self.steps = 1
        self.points_total = 1
        self.converged = False

        self.variances = []
        for v in self.variables.values():
            self.variances.append(v[1])

    def get_next_step(self, log):
            self.next_step = [np.random.normal(x, y)
                              for x, y in zip(self.last_step, self.variances)]
            return self.next_step

    def get_starting_point(self):
            return [np.random.uniform(x[0][0], x[0][1])
                    for x in self.variables.values()]

    def jumpQ(self, log, guess=-1):
            new_point = self.all_data[-1]
            new_lh = self.likelihood(new_point[1])
            if new_lh > self.likelihood_last_step or new_lh/self.likelihood_last_step > np.random.uniform():
                log.info("jumped %i; new_lh: %.2E,  old_lh: %.2E"
                         % (self.steps, new_lh, self.likelihood_last_step))
                log.info("values of obs: " + str(new_point[1]))
                if self.curses:
                    screen.status_mcmc_obs(self.screen, new_point[1])
                self.likelihood_last_step = new_lh
                self.steps = self.steps + 1
                self.last_step = self.next_step
            else:
                log.debug("no jump! new_lh: %.2E,  old_lh: %.2E"
                          % (new_lh, self.likelihood_last_step))
                log.debug("values of obs: " + str(new_point[1]))
            if self.curses:
                screen.status_mcmc(self.screen, self.steps, self.points_total,
                                   self.likelihood_last_step, new_lh)
                if guess > 0:
                    screen.status_mcmc_guess(self.screen, guess, new_lh)

    # Just a simplte criterium as place-holder
    def check_convergence(self):
            if self.steps > 1000:
                self.converged = True

    def run_next_point(self, next, log):
        self.runner.run_point(self, next, os.path.join(self.temp_dir, "id0"),
                              self.output_file, log)
        if self.input_and_observables.empty() is False:
                self.points_total += 1
                try:
                    self.all_data.append(self.input_and_observables.get())
                except:
                    log.error("Problem with getting the observables")

    def test_next(self, log):
        log.info('MCMC count: %i Total points,  %i Jumps'
                 % (self.points_total, self.steps))
        next = self.get_next_step(log)
        self.run_next_point(next, log)
        self.jumpQ(log)
        self.check_convergence()

    def run(self, log):
        # start with random point
        log.info("Starting MCMC")
        self.start_run(log)
        if self.curses:
            screen.start_mcmc(self.screen)
        while self.input_and_observables.empty():
            start = self.get_starting_point()
            self.runner.run_point(self, start,
                                  os.path.join(self.temp_dir, "id0"),
                                  self.output_file, log)
        log.info("Found first point in MCMC: %s" % str(start))
        first_point = self.input_and_observables.get()
        self.all_data.append(first_point)
        self.likelihood_last_step = self.likelihood(first_point[1])
        self.last_step = first_point[0]

        # run the chain
        while not self.converged:
            self.test_next(log)

        self.finish_run(log)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# BROKEN !!!
# Class needs to be updated because of changes in running routine!!!
# I'll do that as soon as all problem with mp are solved
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class MCMC_NN_Scan(MCMC_Scan, ml.NN):
    """Scanner class for MCMC Scans supported by Neural Networks"""

    def __init__(self, screen, input, main_dir, temp_dir, debug, curses, log):
        MCMC_Scan.__init__(self, screen, input, main_dir, temp_dir,
                           debug, curses, log)
        ml.NN.__init__(self)

        self.set_model()

    def get_next_step_nn(self, log):
            log.debug('Guessing next point via NN')
            self.predictor.eval()
            x_try = [
                [np.random.normal(x, y)
                    for x, y in zip(self.last_step, self.variances)]
                for _ in range(1000)
            ]
            y_try = self.predictor(
                Variable(torch.FloatTensor(x_try))).data.cpu().numpy()
            self.next_step = x_try[-1]
            for y, x in zip(y_try, x_try):
                guess_lh = self.likelihood(y)
                if guess_lh > self.likelihood_last_step:
                    self.next_step = x
                    break
            log.debug("proposed point: %s; "
                      + "estimated observables: %s, "
                      + "estimated likelihood: %s"
                      % (self.next_step, y, guess_lh))
            return self.next_step, guess_lh

    def test_next(self, log):
        MIN_POINT = 100  # make it possible to change it via input?!
        INTERVALL = 50

        if self.debug:
            MIN_POINT = 20
            INTERVALL = 20

        log.info('MCMC count: %i Total points,  %i Jumps'
                 % (self.points_total, self.steps))
        if self.points_total <= MIN_POINT:
            next = self.get_next_step(log)
            guess_nn = -1
        else:
            next, guess_nn = self.get_next_step_nn(log)

        if (self.points_total == MIN_POINT) or\
                (self.points_total > MIN_POINT
                 and ((self.points_total - MIN_POINT) % INTERVALL) == 0):
            x = [xx[0] for xx in self.all_data]
            y = [xx[1] for xx in self.all_data]
            self.train(x, y, log)
        self.run_next_point(next, log)
        self.jumpQ(log, guess=guess_nn)
        self.check_convergence()
