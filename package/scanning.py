import json
import os
import shutil
import time
# import datetime
import multiprocessing as mp
import math
# import collections
import itertools
# import numpy as np
# import curses
from six.moves import queue
import importlib

# for the NN
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable

# home-brewed
# from package.aux import LINEAR as LINEAR
# from package.aux import LINEAR_DIFF as LINEAR_DIFF
# from package.aux import LOG as LOG
from package.aux import gauss as gauss
from package.aux import logL as logL
from package.aux import exp_safe as exp_safe
from package.aux import limit_lh as limit_lh

import package.running as running
import xslha
import package.ml as ml
import package.screen as screen

# from aux import dd

# ----------------------------------------------------------
# Scan Class
# All information necessary to 'define' a scan
# ----------------------------------------------------------

class Scan:
    """Main scanner class"""

    def __init__(self, inputs, temp_dir, config):
        # Current xBIT configuration
        self.main_dir = config[0]
        self.screen = config[1]
        self.debug = config[2]
        self.curses = config[3]
        self.log = config[4]
        self.temp_dir = temp_dir

        self.log.info('Initialise scan: %s' % inputs['Setup']['Name'])        

        # main information about the scan
        self.setup = inputs['Setup']
        self.codes = inputs['Included_Codes']
        self.blocks = inputs['Blocks']
        self.observables = inputs['Observables']
        self.variables = inputs['Variables']
        self.ml = inputs['ML']

        self.Short = inputs['Short']

        # Variable to store all commands to execute the programs
        self.run_tools = []


        self.distance_penalty = True

        self.scalings = ["id"] * len(self.observables)

        # for input parameter and values of observables
        self.all_data = []

        # String to Bool
        for c in self.codes:
            self.codes[c] = eval(self.codes[c])

        # load the settings-file
        self.parse_settings(self.setup['Settings'])
        self.set_up_codes()
        self.make_out_dir()
        self.output_file = os.path.join(self.main_dir, "Output",
                                        self.setup['Name'], "SpectrumFiles")

        if inputs['Setup']['Cores'] > 1:
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
        self.runner = running.Runner(self, self.log)

    def parse_settings(self, file):
        """ Parse the settings file which contains
            the paths,  executables,  etc. """
        self.log.info('Parse Settings file: %s' % file)
        with open("Settings/" + file) as json_data:
            d = json.load(json_data)
        self.settings = d
        self.log.debug('Settings: %s' % str(d))

    def set_up_codes(self):
        """ Set up the HEP Tools needed in the scan """
        self.spheno = running.HepTool("SPheno", self.settings['SPheno'],
                                      running.RunSPheno, self.log)
        
        # import all other tools
        new_tools = os.listdir("package/tools")
        for new in new_tools:
            if (new[:2] != "__"):
                new_class = importlib.import_module("package.tools."+new[:-3])
                new_tool = new_class.NewTool()
                if self.codes[new_tool.name]:
                    self.run_tools.append(running.HepTool(new_tool.name,
                                       self.settings[new_tool.name],
                                       new_tool.run, self.log))

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

    def start_run(self):
        self.start_time = time.time()
        self.log.info('Running scan %s' % str(self.setup['Name']))
        if self.curses:
            screen.show_setup(self.screen, self.setup, self.codes)

    def finish_run(self):
        self.log.info('Scan %s finished' % str(self.setup['Name']))
        print("All done!")
        print("Time Needed:               "
              + str(time.time() - self.start_time) + "s")


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# BROKEN !!!
# Class needs to be updated because of changes in running routine!!!
# I'll do that as soon as all problem with mp are solved
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# class MCMC_NN_Scan(MCMC_Scan, ml.NN):
#     """Scanner class for MCMC Scans supported by Neural Networks"""

#     def __init__(self, screen, input, main_dir, temp_dir, debug, curses, log):
#         MCMC_Scan.__init__(self, screen, input, main_dir, temp_dir,
#                            debug, curses, log)
#         ml.NN.__init__(self)

#         self.set_model()

#     def get_next_step_nn(self, log):
#             log.debug('Guessing next point via NN')
#             self.predictor.eval()
#             x_try = [
#                 [np.random.normal(x, y)
#                     for x, y in zip(self.last_step, self.variances)]
#                 for _ in range(1000)
#             ]
#             y_try = self.predictor(
#                 Variable(torch.FloatTensor(x_try))).data.cpu().numpy()
#             self.next_step = x_try[-1]
#             for y, x in zip(y_try, x_try):
#                 guess_lh = self.likelihood(y)
#                 if guess_lh > self.likelihood_last_step:
#                     self.next_step = x
#                     break
#             log.debug("proposed point: %s; "
#                       + "estimated observables: %s, "
#                       + "estimated likelihood: %s"
#                       % (self.next_step, y, guess_lh))
#             return self.next_step, guess_lh

#     def test_next(self, log):
#         MIN_POINT = 100  # make it possible to change it via input?!
#         INTERVALL = 50

#         if self.debug:
#             MIN_POINT = 20
#             INTERVALL = 20

#         log.info('MCMC count: %i Total points,  %i Jumps'
#                  % (self.points_total, self.steps))
#         if self.points_total <= MIN_POINT:
#             next = self.get_next_step(log)
#             guess_nn = -1
#         else:
#             next, guess_nn = self.get_next_step_nn(log)

#         if (self.points_total == MIN_POINT) or\
#                 (self.points_total > MIN_POINT
#                  and ((self.points_total - MIN_POINT) % INTERVALL) == 0):
#             x = [xx[0] for xx in self.all_data]
#             y = [xx[1] for xx in self.all_data]
#             self.train(x, y, log)
#         self.run_next_point(next, log)
#         self.jumpQ(log, guess=guess_nn)
#         self.check_convergence()
