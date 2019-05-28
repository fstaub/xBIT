# import os
import logging
import subprocess
import sys


def new_logger(debug, curses, name, file):
        '''Create Logger'''
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        # file output
        fh = logging.FileHandler(file)
        fh.setLevel(logging.INFO)

        if not curses or debug:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)

        if debug:
            log.setLevel(logging.DEBUG)
            fh.setLevel(logging.DEBUG)
            # show information in screen as wel
            ch.setLevel(logging.ERROR)
            ch.setLevel(logging.DEBUG)

        # format
        formatter = \
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        log.addHandler(fh)
        if debug or not curses:
            formatter = logging.Formatter('%(message)s')
            ch.setFormatter(formatter)
            log.addHandler(ch)
        log.info('Logger %s initialised' % name)
        return log


def command_line_log(command, log):
    log.debug('Terminal command: %s' % command)
    command_line = subprocess.Popen(command, shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
    out, err = command_line.communicate()
    # log.debug(out)
    if err != b'':
        log.error(err.decode("utf-8"))
