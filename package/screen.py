import curses
import time
import datetime

def show_logo(stdscr):
    # update intendations
    curses.resizeterm(150, 45)

    stdscr.clear()
    stdscr.addstr(1, 0, "       __________.______________  ")
    stdscr.addstr(2, 0, "__  __\______   \   \__    ___/   ")
    stdscr.addstr(3, 0, "\  \/  /|    |  _/   | |    |     ")
    stdscr.addstr(4, 0, " >    < |    |   \   | |    |     ")
    stdscr.addstr(5, 0, "/__/\_ \|______  /___| |____|     ")
    # stdscr.addstr(6, 0, "                                  ")
    stdscr.refresh()



def show_setup(stdscr, setup, codes):
    stdscr.addstr(8, 0, "-------------------------------------------------------------------")
    stdscr.addstr(9, 0, "Scan: "+setup['Name'])
    stdscr.addstr(10, 0, "-------------------------------------------------------------------")
    stdscr.addstr(11, 0, "Started:                   "+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d,  %H:%M:%S')))
    stdscr.addstr(12, 0, "Type:                      "+str(setup['Type']))
    # stdscr.addstr(13, 0, "Total number of points:    "+str(setup['Points']))
    stdscr.addstr(14, 0, "Used Cores:                "+str(setup['Cores']))
    stdscr.addstr(15, 0, "Run HiggsBounds:           "+str(codes['HiggsBounds']))
    stdscr.addstr(16, 0, "Run HiggsSignals:          "+str(codes['HiggsSignals']))
    stdscr.addstr(17, 0, "Run Micromegas:            "+str(codes['MicrOmegas']))
    stdscr.refresh()


def update_count(stdscr, notfinished, total):
    x = 22
    ratio = max(min(100, int((100.*(1-1.*notfinished/total)))), 0)
    stdscr.addstr(x, 0, "so far %i of %i points calculated   "
                  % (total-notfinished, total))
    stdscr.addstr(x+1, 0, "[")
    stdscr.addstr(x+1, 101, "]")
    stdscr.addstr(x+1, 1, "#"*ratio)
    stdscr.addstr(x+1, 1+ratio, "-"*(100-ratio))
    stdscr.refresh()


def iteration_nn(stdscr, current, all):
    x = 19
    ratio = int((100.*current/all))
    stdscr.addstr(x, 0, "Iteration %i of %i" % (current, all))
    stdscr.addstr(x+1, 0, "[")
    stdscr.addstr(x+1, 101, "]")
    stdscr.addstr(x+1, 1, "#"*ratio)
    stdscr.addstr(x+1, 1+ratio, "-"*(100-ratio))
    stdscr.refresh()


def status_nn(stdscr, points, loss):
    x = 30
    stdscr.addstr(x, 0, "Status of neutral network                  ")
    stdscr.addstr(x+1, 0, "Data points: %i; Loss: %.2f               "
                  % (points, loss))
    stdscr.refresh()


def train_nn(stdscr, epoch, loss):
    x = 30
    stdscr.addstr(x, 0, "Training the neutral network             ")
    stdscr.addstr(x+1, 0, "Epoch: %i; Loss: %.2f               "
                  % (epoch, loss))
    stdscr.refresh()


def status_mcmc(stdscr, steps, points, lh, last):
    x = 2
    stdscr.addstr(x, 0, "Status of MCMC - Steps %i; Points: %i; Likelihood: %s (last point: %s)"
                  % (steps, points, str(lh), str(last)))
    stdscr.refresh()


def start_mcmc(stdscr):
    x = 26
    stdscr.addstr(x, 0, "Status of MCMC - Searching for starting point")
    stdscr.refresh()


def status_mcmc_obs(stdscr, obs):
    x = 27
    stdscr.addstr(x, 0, "            Values of observables: %s"
                  % str(['{:.2f}'.format(x) for x in obs]))
    stdscr.refresh()


def status_mcmc_guess(stdscr, guess, correct):
    x = 25
    stdscr.addstr(x, 0, "Last point - likelihood guessed by NN: %s; calculated likelihood: %s"
                  % (str(guess), str(correct)))
    stdscr.refresh()


def current_point_core(stdscr, input, core):
    x = 35
    stdscr.addstr(x+core, 0, "Core %s - current point %s               "
                  % (core+1, str(['{:.2f}'.format(x) for x in input])))
    stdscr.refresh()
