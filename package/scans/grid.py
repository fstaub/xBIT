from package.scanning import Scan as Scan
import itertools

from package.aux import LINEAR as LINEAR
from package.aux import LINEAR_DIFF as LINEAR_DIFF
from package.aux import LOG as LOG


scan_name = "Grid"

class NewScan(Scan):
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
