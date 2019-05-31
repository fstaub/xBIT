# standard packages
import os
import xslha
import shutil
import multiprocessing as mp

# modules of xBIT
import package.screen as screen
import package.debug  as debug

# ----------------------------------------------------------
# Class to run the different codes
# ----------------------------------------------------------


class HepTool:
    """Main class to run the different HEP tools"""

    def __init__(self, name, settings, runner, log):
        log.info('Class %s initialised' % name)
        self.name = name
        self.settings = settings
        self.runner = runner
        self.log = log

    def run(self, spc_file, temp_dir, log):
        log.info('Running %s ' % self.name)
        os.chdir(temp_dir)

        try:
            self.runner(self.settings, spc_file, temp_dir, log)
        except Exception as e:
            print("Problem in running ", self.name)
            print(e)

# -----------------------------
#  Auxiliary Run and Parse functions for different codes
# -----------------------------


def RunSPheno(settings, spc_file, dir, log):
    if os.path.exists(settings['OutputFile']):
        os.remove(settings['OutputFile'])
    debug.command_line_log(settings['Command'] + " " + settings['InputFile'], log)

class Runner():
    def __init__(self, scan, log):
        log.info('Initialise runner class. Number of cores: %s'
                 % str(scan.inputs['Setup']['Cores']))

        self.scan = scan

        # self.all_valid = []
        # self.all_invalid = []
        # self.all_data = []

        # create temporary directories
        for x in range(scan.inputs['Setup']['Cores']):
            os.makedirs(os.path.join(scan.config.temp_dir, "id" + str(x)))

        # setup_loggers:
        self.loggers = [
            debug.new_logger(scan.config.debugQ, scan.config.cursesQ, "id" + str(x),
                             os.path.join(scan.config.temp_dir, "id" + str(x))
                             + "/id" + str(x) + ".log")
            for x in range(scan.inputs['Setup']['Cores'])
        ]

    def run(self, log, sample=[]):
        if len(sample) < 1:
            self.all_parameter_variables = self.scan.generate_parameter_points(
                self.scan.inputs['Variables'], self.scan.inputs['Setup']['Points']
            )
        else:
            self.all_parameter_variables = sample

        # move points into queue in order to distribute work on several cores
        for x in self.all_parameter_variables:
            self.scan.all_points.put(x)

        if self.scan.inputs['Setup']['Cores'] > 1:
            self.multicore(log)
        else:
            self.singlecore(log)

    def multicore(self, log):
        # sample is not empty in case that the NN proposes the next points
        log.info('Starting multcore module. Number of cores: %s'
                 % str(self.scan.inputs['Setup']['Cores']))
        with mp.Manager() as manager:
            List_all = manager.list()
            List_valid = manager.list()
            List_invalid = manager.list()

            # define the processes and let them run
            processes = [mp.Process(
                target=self.run_all_points,
                args=(self.scan,
                      os.path.join(self.scan.config.temp_dir, "id" + str(x)),
                      x, self.loggers[x],
                      List_all, List_valid, List_invalid))
                for x in range(self.scan.inputs['Setup']['Cores'])]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            self.scan.all_data = self.scan.all_data + list(List_all)
            self.scan.all_valid = self.scan.all_valid + list(List_valid)
            self.scan.all_invalid = self.scan.all_invalid + list(List_invalid)

    def singlecore(self, log):
        self.run_all_points(self.scan,
                            os.path.join(self.scan.config.temp_dir, "id0"), 0,
                            self.loggers[0],
                            self.scan.all_data,
                            self.scan.all_valid,
                            self.scan.all_invalid
                            )

    # run points
    def run_all_points(self, scan, dir, nr, log, l1=None, l2=None, l3=None):
        log.info("Started with running the points")
        while not scan.all_points.empty():

            # 'progress-bar'
            if nr == 0:
                if scan.config.cursesQ:
                    screen.update_count(scan.config.screen, scan.all_points.qsize(),
                                        scan.inputs['Setup']['Points'])
                else:
                    log.info("")
                    log.info("%i Points of %i Points left"
                             % (scan.all_points.qsize(), scan.inputs['Setup']['Points']))

            # running a point
            try:
                # in order to make sure that the last element
                # hasn't been 'stolen' in the meantime by another core
                point = scan.all_points.get()
                self.run_point(scan, point, dir, scan.output_file + str(nr),
                               l1, l2, l3, log)
            except:
                # break
                continue  # maybe, there was another problem than an empty queue?!
                          # let's try to continue instead of stopping

        if scan.config.cursesQ:
            screen.update_count(scan.config.screen, 0, scan.inputs['Setup']['Points'])

    def bad_point_check(self, scan, log):
        if scan.inputs['Setup']['Interrupt'][0] == "True":
            values = []
            spc = xslha.read(scan.settings['SPheno']['OutputFile'])
            for obs in scan.inputs['Observables'].values():
                try:
                    values.append(spc.Value(obs['SLHA'][0], obs['SLHA'][1]))
                except:
                    values.append(obs['MEAN'])
            if scan.likelihood(values) < scan.inputs['Setup']['Interrupt'][1]:  # likelihood too small
                log.info('Stopping further calculations for this point'
                         + ' because of bad likelihood')
                return True
            else:
                return False
        else:
            return False

    def run_point(self, scan, point, temp_dir, output_file,
                  list_all, list_valid, list_invalid, log):
        log.info('Running point with input parameters: %s'
                 % str(point))
        if scan.config.cursesQ:
            screen.current_point_core(scan.config.screen, point, int(temp_dir[-1]))
        scan.write_lh_file(point, temp_dir, scan.settings['SPheno']['InputFile'])
        scan.spheno.run(scan.settings['SPheno']['OutputFile'], temp_dir, log)
        if self.bad_point_check(scan, log):
            return
        if os.path.exists(scan.settings['SPheno']['OutputFile']):
            log.info('SPheno spectrum produced')
            for run_now in scan.run_tools:
                try:
                    run_now.run(scan.settings['SPheno']['OutputFile'], temp_dir, log)
                except Exception as e:
                    print(e)

            if scan.Short:
                spc = xslha.read(scan.settings['SPheno']['OutputFile'])
                debug.command_line_log("echo " + ' '.join(map(str,point)) + ' ' + ' '.join(map(str,[spc.Value(obs['SLHA'][0], obs['SLHA'][1]) for obs in scan.inputs['Observables'].values()])) + " >> " + output_file, log)
            else:
                debug.command_line_log("cat " + scan.settings['SPheno']['OutputFile']
                                   + " >> " + output_file, log)
                debug.command_line_log("echo \"ENDOFPARAMETERPOINT\" >> "
                                   + output_file, log)
                
            if len(scan.inputs['Observables']) > 0:
                log.info('Reading spectrum file')
                spc = xslha.read(scan.settings['SPheno']['OutputFile'])
                try:
                    list_all.append(
                        [point, [spc.Value(obs['SLHA'][0], obs['SLHA'][1])
                                 for obs in scan.inputs['Observables'].values()]])
                    list_valid.append(point)
                    log.debug("Observables: %s"
                              % str([spc.Value(obs['SLHA'][0], obs['SLHA'][1])
                                    for obs in scan.inputs['Observables'].values()]))
                except:
                    list_invalid.append(point)
                    log.warning('Observable(s) missing in SLHA file')
        else:
            list_invalid.append(point)
            
            # We keep the non-valid points for plotting
            if not scan.Short:
                debug.command_line_log("cat " + scan.settings['SPheno']['InputFile']
                                   + " >> " + output_file, log)
                debug.command_line_log("echo \"ENDOFPARAMETERPOINT\" >> "
                                  + output_file, log)
            log.info('NO SPheno spectrum produced')
