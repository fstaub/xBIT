from package.scanning import Scan as Scan
import itertools
import numpy as np

scan_name = "Grid"

class NewScan(Scan):
    """Scanner class for Grid Scans"""

    def __init__(self, inputs, config):
        Scan.__init__(self, inputs, config)

    def run(self):
        self.runner.run(self.config.log, sample=[])

    def generate_parameter_points(self, vars, points):
        all = []
        for x in vars:
            all.append(eval(vars[x]))
        temp = list(itertools.product(*all))
        all = [list(xx) for xx in temp]
        self.setup['Points'] = len(all)
        return all       
