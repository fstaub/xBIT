import os
import shutil
import package.debug  as debug

class NewTool():
    def __init__(self):
        self.name = "Vevacious" 

    def run(self,path, bin, input, output, spc_file, dir, log):
        try:
            shutil.copy(input, path)
            os.chdir(path)
            debug.command_line_log(bin, log)
            shutil.copyfile(input, os.path.join(dir, input))
            os.chdir(dir)
        except Exception as e: 
            log.error("Problem occured running Vevacious!")        
            log.error(e)
