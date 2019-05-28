from package.scanning import Scan as Scan

scan_name = "Random"

class NewScan(Scan):
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
