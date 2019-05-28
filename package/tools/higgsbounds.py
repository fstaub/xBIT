import os
import shutil
import package.debug  as debug

class NewTool():
    def __init__(self):
        self.name = "HiggsBounds" 

    def run(self,path, bin, input, output, spc_file, dir, log):
        debug.command_line_log(path + bin + " " + input + " " + dir + "/", log)
        # Reading the Output file
        if os.path.exists(bin + "_results.dat"):
            for line in open(bin + "_results.dat"):
                li = line.strip()
                if not li.startswith("#"):
                    hb_res = list(filter(None, line.rstrip().split(' ')))
            # Append output to the SPheno file
            debug.command_line_log("echo \"Block " + bin.upper() + " # \" >> "
                                + spc_file, log)
            for i in range(1, len(hb_res)):
                debug.command_line_log("echo \"" + str(i) + " " + str(hb_res[i])
                                    + " # \" >> " + spc_file, log)
        else:
            log.error("HiggsBounds output not written!",
                    path + bin + " " + input + " " + dir)