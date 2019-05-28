from package.scanning import Scan as Scan
import itertools

from package.aux import LINEAR as LINEAR
from package.aux import LINEAR_DIFF as LINEAR_DIFF
from package.aux import LOG as LOG


scan_name = "Grid"

class NewScan(Scan):
    """Scanner class for Grid Scans"""

    def __init__(self, inputs, temp_dir, config):
        Scan.__init__(self, inputs, temp_dir, config)

    def run(self):
        self.start_run()
        self.runner.run(self.log, sample=[])
        self.finish_run()

    def generate_parameters(self, vars, points):
        all = []
        for x in vars:
            all.append(eval(vars[x]))
        temp = list(itertools.product(*all))
        all = [list(xx) for xx in temp]
        self.setup['Points'] = len(all)
        return all       
