import os
import shutil
import package.debug  as debug

class NewTool():
    def __init__(self):
        self.name = "HiggsSignals" 

    def run(self, settings, spc_file, temp_dir, log):
        # Settings
        command = settings['Command']
        options = settings['Options']
        output_file = settings['OutputFile']

        # Clean up
        if os.path.exists(output_file):
            os.remove(output_file)


        # Run HiggsBounds
        debug.command_line_log(command + " " + options + " " + temp_dir + "/", log)

        # Reading the Output file
        if os.path.exists(output_file):
            for line in open(output_file):
                li = line.strip()
                if not li.startswith("#"):
                    results = list(filter(None, line.rstrip().split(' ')))
            # Append output to the SPheno file
            debug.command_line_log("echo \"Block " + self.name.upper() + " # \" >> "
                                + spc_file, log)
            for i in range(1, len(results)):
                debug.command_line_log("echo \"" + str(i) + " " + str(results[i])
                                    + " # \" >> " + spc_file, log)
        else:
            log.error("HiggsSignals output not written!",
                    command + " " + options + " " + temp_dir)