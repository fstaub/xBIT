from package.scanning import Scan as Scan

scan_name = "Random"

class NewScan(Scan):
    """Scanner class for Random Scans"""

    def __init__(self, inputs, temp_dir, config):
        Scan.__init__(self, inputs, temp_dir, config)
    
    def run(self):
            self.start_run()
            self.runner.run(self.log, sample=[])
            self.finish_run()

    def generate_parameters(self, vars, points):
        all = []
        for _ in range(points):
            all += [[eval(vars[xx]) for xx in vars]]
        return all
