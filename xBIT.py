"""
Tool for parameter scans in high-energy-physics

In order to run a scan, use

     python3 xBIT.py Input/inputfile option

All information about the scan are defined in the inputfile.
The following options can be set:
  -- debug: print debug messages
  -- clean: clean up the temporary directory before running
"""




import argparse
import os
import time
import sys
# import logging
import shutil
import json
import collections
import curses
import re

sname=str(sys.argv[0]) 
mainpath = re.search("^.*\/",sname)
if mainpath:
    sys.path.append(mainpath.group(0))

# home-brewed
# import running
import scanning

# import ml
import debug
import screen




# check for non-standard packages
try:
    import xslha
except:
    print("Error: the module xslha is needed")
    print("you can install it via:")
    print("   pip install xslha")
    print("and run again")
    sys.exit()


try:
    import torch
except:
    print("Error: pytorch is needed")
    print("You find details about the installation here: https://pytorch.org/")
    sys.exit()

# to-do/questions:
# -definition/format of input (yaml instead of json?)




# Read the input file and extract all necessary information
def parse_input(file, log):
    log.info('Parsing input file: %s' % file)
    with open(file) as json_data:
        d = json.load(json_data, object_pairs_hook=collections.OrderedDict)
    log.debug('Content of input file: %s' % str(d))
    return d


####################################################################################
# MAIN PROGRAM
####################################################################################

# -------------------------------------
# Initialisation
# -------------------------------------

# Try to open input file
try:
    parser = argparse.ArgumentParser(
        description='Please give the name of the input file.')
    parser.add_argument('inputfiles',
                        metavar='File', type=str,
                        nargs='+', help='an integer for the accumulator')
    parser.add_argument("--short", help="Store output in short form",
                        action="store_true")
    parser.add_argument("--debug", help="write debug information",
                        action="store_true")
    parser.add_argument("--clean", help="clean temporary directory",
                        action="store_true")
    parser.add_argument("--quiet", help="use minimal screen output",
                        action="store_true")
    parser.add_argument("--Name", type=str, help='name of the scan')
    parser.add_argument("--DensityPenality", type=bool,
                        help='penaluity for points too close')
    parser.add_argument("--TrainLH", type=bool, help='train the likelihood')
    parser.add_argument("--Classifier", type=bool,
                        help='Include classifier for valid/invalid points')
    parser.add_argument("--Points", type=int, help='number of points')
    parser.add_argument("--Cores", type=int, help='number of cores')
    parser.add_argument("--Iterations", type=int, help='number of iterations')
    parser.add_argument("--Epochs", type=int,
                        help='Maximal numbers of epochs in training')
    parser.add_argument("--LR", type=float,
                        help='Learning Rate for Neutral Network')
    parser.add_argument("--Neurons", type=str,
                        help='Number of neurons in hidden layer')
    parser.add_argument("--curses",
                        help="use curses to for nice screen output",
                        action="store_true")
    args = parser.parse_args()
except:
    print("Please give an input file")

# set up paths
cwd = os.getcwd()

if args.clean:
    print("Cleaning temporary directory")
    shutil.rmtree(os.path.join(cwd, "Temp"))
    os.makedirs(os.path.join(cwd, "Temp"))

timestamp = str(time.time()).replace(".", "")
temporary_dir = os.path.join(cwd, "Temp", timestamp)
os.makedirs(temporary_dir)

main_logger = debug.new_logger(args.debug, args.curses,
                               "main", temporary_dir+"/xBIT.log")
main_logger.info('Starting xBIT with then following arguments: %s' % args)


def set_default(input, log):
    # Observables
    if 'Observables' not in input:
        input['Observables'] = {}

    # Setting of NN
    if 'ML' not in input:
        input['ML'] = {}
    else:
        if 'Epochs' not in input['ML']:
            input['ML']['Epochs'] = 5000
        if 'Neurons' not in input['ML']:
            input['ML']['Neurons'] = [50, 50, 50]
        if 'LR' not in input['ML']:
            input['ML']['LR'] = 0.001
        if 'DensityPenality' not in input['ML']:
            input['ML']['DensityPenality'] = False
        else:
            input['ML']['DensityPenality'] = \
                eval(input['ML']['DensityPenality'])
        if 'Classifier' not in input['ML']:
            input['ML']['Classifier'] = True
        else:
            input['ML']['Classifier'] = eval(input['ML']['Classifier'])
        if 'TrainLH' not in input['ML']:
            input['ML']['TrainLH'] = False
        else:
            input['ML']['TrainLH'] = eval(input['ML']['TrainLH'])

    # General settings
    if 'Interrupt' not in input['Setup']:
        input['Setup']['Interrupt'] = [False, 0]
    if 'Iterations' not in input['Setup']:
        input['Setup']['Iterations'] = 100

    log.debug('Content of input after applying default settings: %s'
              % str(input))
    return input


def check_terminal_arguments(input, args, log):
    output = input
    if args.short:
        output['Short'] = True
    else:
        output['Short'] = False
    if args.Name:
            output['Setup']['Name'] = args.Name
    if args.Points:
            output['Setup']['Points'] = args.Points
    if args.Cores:
            output['Setup']['Cores'] = args.Cores
    if args.Iterations:
            output['Setup']['Iterations'] = args.Iterations
    if args.LR:
            output['ML']['LR'] = args.LR
    if args.Epochs:
            output['ML']['Epochs'] = args.Epochs
    if args.DensityPenality:
            input['ML']['DensityPenality'] = args.DensityPenality
    if args.Classifier:
            input['ML']['Classifier'] = args.Classifier
    if args.TrainLH:
            input['ML']['TrainLH'] = args.TrainLH
    if args.Neurons:
            output['ML']['Neurons'] = eval(args.Neurons)
    log.debug('Content of input with terminal input: %s' % str(output))
    return output

# ----------------------------------------------------------------------------
# looping over the input files and running scans
# ----------------------------------------------------------------------------


def main(stdscr, debug, curses):
    if curses:
        stdscr.clear()
        screen.show_logo(stdscr)

    for i in args.inputfiles:
        input = parse_input(i, main_logger)
        input = set_default(input, main_logger)
        input = check_terminal_arguments(input, args, main_logger)

        if input['Setup']['Type'] == "Grid":
            scan = scanning.Grid_Scan(stdscr, input, cwd,
                                      temporary_dir, debug, curses,
                                      main_logger)
        elif input['Setup']['Type'] == "Random":
            scan = scanning.Random_Scan(stdscr, input, cwd,
                                        temporary_dir, debug, curses,
                                        main_logger)
        elif input['Setup']['Type'] == "MCMC":
            scan = scanning.MCMC_Scan(stdscr, input, cwd,
                                      temporary_dir, debug, curses,
                                      main_logger)
        elif input['Setup']['Type'] == "MLS":
            scan = scanning.MLS_Scan(stdscr, input, cwd,
                                     temporary_dir, debug, curses,
                                     main_logger)
        elif input['Setup']['Type'] == "MCMC_NN":
            scan = scanning.MCMC_NN_Scan(stdscr, input, cwd,
                                         temporary_dir, debug, curses,
                                         main_logger)
        else:
            main_logger.error("Scan Type %s not defined" % input['Type'])

        scan.run(main_logger)

    if curses:    # in order to keep the last information on the screen
        mypad_contents = []
        for i in range(0, 25):
            mypad_contents.append(stdscr.instr(i, 0))
        return mypad_contents


if args.curses and not args.debug:
    # fancy screen output using curses
    screenout = curses.wrapper(main, args.debug, args.curses)
    for i in screenout:
        print(i.decode("utf-8"))
else:
    # standard screen output for the debug mode
    print("       __________.______________  ")
    print("__  __\______   \   \__    ___/   ")
    print("\  \/  /|    |  _/   | |    |     ")
    print(" >    < |    |   \   | |    |     ")
    print("/__/\_ \|______  /___| |____|     ")
    print("                                  ")
    main("nix", args.debug, args.curses)
