import os
import shutil
import xslha
import package.debug  as debug

class NewTool():
    def __init__(self):
        self.name = "MicrOmegas" 

    def run(self,settings, spc_file, temp_dir, log):
        # Settings
        command = settings['Command']
        output_file = settings['OutputFile']
        dark_matter = settings['DM_Candidate']

        # Check the LSP
        spc = xslha.read(spc_file)
        if spc.Value('LSP', [1]) == dark_matter:   
            # Clean up old results
            if os.path.exists(output_file):
                os.remove(output_file)
  
            # Run MO
            debug.command_line_log(command, log)

            # Glue output to the SPheno file
            if os.path.exists(output_file):
                debug.command_line_log("echo \"Block DARKMATTER # \" >> "
                                    + os.path.join(temp_dir, spc_file), log)
                debug.command_line_log("cat " + output_file + " >> "
                                    + os.path.join(temp_dir, spc_file), log)
            else:
                log.error("MicrOmegas output not written!")