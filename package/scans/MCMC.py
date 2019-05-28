from package.scanning import Scan as Scan

scan_name = "MCMC"

class NewScan(Scan):
    """Scanner class for Basic MCMC Scans:
       A Marcov-Chain-Monte-Carlo class (MCMC) based
       on the Metropolis algorithm
    """

# to-do/questions:
#  - define convergence criterium
#  - check that tested points are not outside the scan ranges?
#  - let several chains run in parallel ( = > how to define number of chains in input?)
#  - gauss distribution always assumed  = > sufficient?

    def __init__(self, screen, input, main_dir, temp_dir, debug, curses, log):
        Scan.__init__(self, screen, input, main_dir, temp_dir,
                      debug, curses, log)
        self.last_step = []
        self.likelihood_last_step = 0
        self.steps = 1
        self.points_total = 1
        self.converged = False

        self.variances = []
        for v in self.variables.values():
            self.variances.append(v[1])

    def get_next_step(self, log):
            self.next_step = [np.random.normal(x, y)
                              for x, y in zip(self.last_step, self.variances)]
            return self.next_step

    def get_starting_point(self):
            return [np.random.uniform(x[0][0], x[0][1])
                    for x in self.variables.values()]

    def jumpQ(self, log, guess=-1):
            new_point = self.all_data[-1]
            new_lh = self.likelihood(new_point[1])
            if new_lh > self.likelihood_last_step or new_lh/self.likelihood_last_step > np.random.uniform():
                log.info("jumped %i; new_lh: %.2E,  old_lh: %.2E"
                         % (self.steps, new_lh, self.likelihood_last_step))
                log.info("values of obs: " + str(new_point[1]))
                if self.curses:
                    screen.status_mcmc_obs(self.screen, new_point[1])
                self.likelihood_last_step = new_lh
                self.steps = self.steps + 1
                self.last_step = self.next_step
            else:
                log.debug("no jump! new_lh: %.2E,  old_lh: %.2E"
                          % (new_lh, self.likelihood_last_step))
                log.debug("values of obs: " + str(new_point[1]))
            if self.curses:
                screen.status_mcmc(self.screen, self.steps, self.points_total,
                                   self.likelihood_last_step, new_lh)
                if guess > 0:
                    screen.status_mcmc_guess(self.screen, guess, new_lh)

    # Just a simplte criterium as place-holder
    def check_convergence(self):
            if self.steps > 1000:
                self.converged = True

    def run_next_point(self, next, log):
        self.runner.run_point(self, next, os.path.join(self.temp_dir, "id0"),
                              self.output_file, log)
        if self.input_and_observables.empty() is False:
                self.points_total += 1
                try:
                    self.all_data.append(self.input_and_observables.get())
                except:
                    log.error("Problem with getting the observables")

    def test_next(self, log):
        log.info('MCMC count: %i Total points,  %i Jumps'
                 % (self.points_total, self.steps))
        next = self.get_next_step(log)
        self.run_next_point(next, log)
        self.jumpQ(log)
        self.check_convergence()

    def run(self, log):
        # start with random point
        log.info("Starting MCMC")
        self.start_run(log)
        if self.curses:
            screen.start_mcmc(self.screen)
        while self.input_and_observables.empty():
            start = self.get_starting_point()
            self.runner.run_point(self, start,
                                  os.path.join(self.temp_dir, "id0"),
                                  self.output_file, log)
        log.info("Found first point in MCMC: %s" % str(start))
        first_point = self.input_and_observables.get()
        self.all_data.append(first_point)
        self.likelihood_last_step = self.likelihood(first_point[1])
        self.last_step = first_point[0]

        # run the chain
        while not self.converged:
            self.test_next(log)

        self.finish_run(log)