import os
import shutil
import package.debug  as debug

class NewTool():
    def __init__(self):
        self.name = "MicrOmegas" 

    def run(self,path, bin, input, output, spc_file, dir, log):
        debug.command_line_log(path + bin, log)
        # Glue output to the SPheno file
        if os.path.exists(output):
            debug.command_line_log("echo \"Block DARKMATTER # \" >> "
                                + os.path.join(dir, spc_file), log)
            debug.command_line_log("cat " + output + " >> "
                                + os.path.join(dir, spc_file), log)
        else:
            log.error("MicrOmegas output not written!")