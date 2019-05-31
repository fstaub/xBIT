from package.scanning import Scan as Scan
import numpy as np

scan_name = "Random"

class NewScan(Scan):
    """Scanner class for Random Scans"""

    def __init__(self, inputs, config):
        Scan.__init__(self, inputs, config)
    
    def run(self):
            self.runner.run(self.config.log, sample=[])

    def generate_parameter_points(self, vars, points):
        all = []
        for _ in range(points):
            all += [[eval(vars[xx]) for xx in vars]]
        return all
