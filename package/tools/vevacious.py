import os
import shutil
import package.debug  as debug

class NewTool():
    def __init__(self):
        self.name = "Vevacious" 

    def run(self, settings, spc_file, temp_dir, log):
        path = settings['Path']
        command = setting['Command']

        # copy spc file to Vevacious directory
        shutil.copy(spc_file, path)

        # Run Vevacious
        os.chdir(path)
        debug.command_line_log(command, log)

        # copy spc file with Vevacious results back to temporary dir
        shutil.copyfile(spc_file, os.path.join(temp_dir, spc_file))
        os.chdir(temp_dir)
