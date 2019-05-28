from package.scanning import Scan as Scan
from package.ml import NN as NN
import itertools

from package.aux import LINEAR as LINEAR
from package.aux import LINEAR_DIFF as LINEAR_DIFF
from package.aux import LOG as LOG


scan_name = "MLS"

class NewScan(Scan, NN):
    """Scanner class for Machine Learning Scans

       Algorithm to improve parameter space sampling using a neural network
       Basic idea:
         1) Start with random sample
         2) Train NN
         3) Use NN to propose new points with a 'good likelihood'
        Approach proposed by from J. Ren,  L. Wu,  J.M. Yang,  J. Zhao
    """

    def __init__(self, screen, input, main_dir, temp_dir, debug, curses, log):
        Scan.__init__(self, screen, input, main_dir, temp_dir,
                      debug, curses, log)
        NN.__init__(self)

    def generate_parameters(self, vars, points):
        all = []
        for _ in range(points):
            all += [[eval(vars[xx]) for xx in vars]]
        return all

    def run(self, log):
        log.info("\n\n##### Iteration: %i #####" % (1))
        if self.curses:
            screen.iteration_nn(self.screen, 1, self.iterations)
        # first run: just a ordinary random scan

        self.start_run(log)
        save_points = self.setup['Points']
        self.setup['Points'] = 5 * self.setup['Points']
        self.runner.run(log, sample=[])
        self.setup['Points'] = save_points

        while not self.input_and_observables.empty():
            self.all_data.append(self.input_and_observables.get())
        while not self.valid_points.empty():
            self.all_valid.append(self.valid_points.get())
        while not self.invalid_points.empty():
            self.all_invalid.append(self.invalid_points.get())

        # iterate
        for i in range(1, self.iterations):
            x = [xx[0] for xx in self.all_data]
            y = [xx[1] for xx in self.all_data]
            if self.train_lh:
                y_scale = [[math.log(self.likelihood(yy) + 1.0e-10)]
                           for yy in y]
            else:
                y_scale = self.scale_data(y, log)

            self.set_predictor()
            self.train(self.predictor, x, y_scale, log)

            if self.use_classifier:
                self.set_classifier()
                x_class = self.all_valid + self.all_invalid[0:min([len(self.all_valid),len(self.all_invalid)-1])]
#                print("valid, invalid:", len(self.all_valid), len(self.all_invalid))
                y_class = [[0.99, 0.01] for j in self.all_valid] + \
                          [[0.01, 0.99] for j in self.all_invalid[0:min([len(self.all_valid),len(self.all_invalid)-1])]]
                self.train(self.classifier, x_class, y_class, log)

            proposed_points = self.new_points(
                self.setup['Points'],
                #len(self.variables)**2 * (i + 1) * self.setup['Points']
                len(self.variables)**2 * self.setup['Points']                
            )
            log.info("##### Iteration: %i #####" % (i + 2))
            if self.curses:
                screen.iteration_nn(self.screen, i + 1, self.iterations)
            self.runner.run(log, sample=proposed_points)
            while not self.input_and_observables.empty():
                self.all_data.append(self.input_and_observables.get())

        self.finish_run(log)

    def guess_LH(self, x_in):
        # Run predictor
        if self.train_lh:
            y_try = self.predictor(
                Variable(torch.FloatTensor(x_in))).data.cpu().numpy()
        else:
            y_try = self.rescale(self.predictor(
                Variable(torch.FloatTensor(x_in))).data.cpu().numpy())

        # Run classifier, if included
        if self.use_classifier:
            y_class = self.classifier(
                Variable(torch.FloatTensor(x_in))).data.cpu().numpy()
#            print("classifier result: ", y_class)
        else:
            y_class = [[0.99, 0.01] for _ in y_try]

        # Filter valid points and calculate likelihood
        lh_valid = []
        x_valid = []
        for y, x, yc in zip(y_try, x_in, y_class):
            if (yc[0] > yc[1]):
                if self.train_lh:
                    lh_valid.append(exp_safe(y))
                else:
                    lh_valid.append(self.likelihood(y))
                x_valid.append(x)

        # Calculate final likelihood (optionally with distance penalty)
        if not self.distance_penalty:
            likelihood = lh_valid
        else:
            likelihood = self.min_Delta_LH(lh_valid, x_valid)

        return likelihood, x_valid

    def new_points(self, good_points, points):
        # estimate new points with good likelihood
        self.predictor.eval()
        best_lh=[]
        increase=1
        while (len(best_lh)<5):
            x_try = self.generate_parameters(self.variables, max(increase*points,100000))
            likelihood, x_try = self.guess_LH(x_try)
            best_lh = [l for l in likelihood if l < 1]
            self.log.debug("Proposed LHs: %s" % (str(best_lh)))
            increase += 1
            if increase > 100:
                break
        best_lh = max(best_lh)
        count = 1
        count_all = 0
        self.log.info("Best LH proposed by NN: %s" % (str(best_lh)))
        while count < int(0.9 * good_points):
            count_all = count_all + 1            
            for l, x in zip(likelihood, x_try):
                if 2. > l > limit_lh(count_all) * best_lh:  # 2 > might be necessary for the LH fit because some good points are predicted to have slighlyt higher LH;
                    self.log.debug("Accepted point %i by NN: LH: %s"
                                   % (count, str(l)))
                    if count == 1:
                        new_good_points = [x]
                    else:
                        new_good_points = np.concatenate((new_good_points, [x]))
                    count = count + 1
                    if count > int(0.9 * good_points):
                        break
            self.log.debug("Accepted Points by NN: %i after %i iterations" % (count, count_all))            
            x_try = self.generate_parameters(self.variables, max(increase*points,100000))            
            likelihood, x_try = self.guess_LH(x_try)
            # print("x_try", likelihood)

        # add 10% random points
        new_x = np.concatenate((
            self.generate_parameters(self.variables, int(0.1 * good_points)),
            new_good_points
        ))
        return new_x

    def min_Delta_LH(self, lh, xval):
        '''Function to calculate the minimal 'distance' of a new point
        to previously sampled points'''
        all_points = np.array([x[0] for x in self.all_data])
        norm = np.array([eval(x.replace("LINEAR", "LINEAR_DIFF"))
                         for x in self.variables.values()])
        lh_out = []
        for l, x in zip(lh, xval):
            min_d = np.amax(np.abs((all_points - x) / norm),
                            axis=1).min()
            lh_out.append(l * min_d**2)
        return lh_out     
