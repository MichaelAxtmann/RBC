import os.path
import subprocess
import json
import re
import sys
import os

def blub():
    print "blub"

def add_header_commands(dir):
    try:
        with open(dir + "/compile_commands.json", "r") as read_file:
            data = json.load(read_file)
    except IOError:
        sys.stderr.write("File with compile commands not found.\n")
        sys.exit(os.EX_IOERR)
    
    comp_unit_dict = {}
    for compile_unit in data:
        comp_unit_dict[compile_unit["file"]] = compile_unit

    output_dict = comp_unit_dict.copy()

    for unit in comp_unit_dict:
        command = comp_unit_dict[unit]["command"].encode('ascii', 'ignore')
        # Remove -o command as the folder may not exist yet.
        # -MMD Print user include paths
        # -MF- Print result to stdout
        # -E   Just do preprocessing -- much faster
        command_mod = re.sub(" -o [^ ]+ ", " -o /dev/null ", command) + " -MMD -MF- -E"
        try:
            string = subprocess.check_output(command_mod, shell=True)
        except subprocess.CalledProcessError as e:
            sys.stderr.write("Warning: Unable to read header includes.\n" + command_mod + "\n")
            continue
        # We get an empty string if no headers are included at all.
        # We also might get an empty string if we use gcc. In this
        # case, we try to print the output to a tmp file.
        if not string:
            command_mod = re.sub(" -o [^ ]+ ", " -o /dev/null ", command) + " -MMD -MFpreprocessor.log -E"
            try:
                subprocess.check_output(command_mod, shell=True)
            except subprocess.CalledProcessError as e:
                sys.stderr.write("Warning: Unable to write header includes.\n" + command_mod + "\n")
                continue
            if os.path.isfile("./preprocessor.log"):
                with open('./preprocessor.log', 'r') as myfile:
                    string = myfile.read()
            else:
                sys.stderr.write("Warning: Unable to read stored header includes from preprocessor.log.\n")
        if not string:
            sys.stderr.write("Warning: Unable to generate header includes.\n" + command_mod + "\n")
            continue
        # Remove output file
        # Even though we use /dev/null as output of the object file, gcc
        # sometimes still prints <cpp-file-name>.o ...
        string = re.sub("/dev/null: ", "", string)
        string = re.sub(".*\.o:", "", string)
        # Create spaces between header files
        string = re.sub("(\\\n)|(\\\\\n)", "", string)
        # Remove suffix and prefix of spaces -- avoids empty string after split
        string = string.strip()
        # Split string into list of header files
        headers = re.split(" +", string)
        # Process all header files of this compilation unit.
        # First string in list of headers is actually the source file
        if len(headers) == 0:
            sys.stderr.write("Warning: Description of compilation unit has unexpected format.\n" + json.dumps(comp_unit_dict[unit]) + "\n")
            continue
        source = headers[0]
        # Compile commands of the source file
        cc_source = output_dict[headers[0]]
        for header in headers[1:]:
            real_header = os.path.realpath(header)
            # Break if compilation commands already exist for this header
            if real_header not in output_dict:
                # The source file uses the compile commands of the including source file
                # but we set the file name to the header file name.
                cc_header = cc_source.copy()
                # Note: The command still uses the source file for the -c and -o flags.
                # However, this parameters are not used by youcompleteme.
                # cc_header["command"] = cc_header["command"].replace("-c " + source, "-c " + header)
                cc_header["file"] = real_header
                # print "\tFile added: " + header
                # Add new compile command description to our output
                output_dict[real_header] = cc_header

    use_clang_cpp_for_mpi = 1
    for unit in output_dict:
        command = output_dict[unit]["command"].encode('ascii', 'ignore')
        # Replace MPI wrapper to get MPI include dirs and libraries
        command_list = command.split(" ", 1)
        if len(command_list) == 2:
            if "mpi" in command_list[0]:
                try:
                    mpi_command = subprocess.check_output(command_list[0] + " -show", shell=True)
                except subprocess.CalledProcessError as e:
                    sys.stderr.write("Warning: Unable to resolve MPI wrapper "
                                     + command_list[0]
                                     + "\n")
                    continue
                if use_clang_cpp_for_mpi:
                    splitted_command = mpi_command.split(" ", 1)
                    mpi_command = "clang++ "
                    # Add prefix for the case that includes,... exist.
                    if len(splitted_command) == 2:
                        mpi_command += splitted_command[1]
                print "test"
                output_dict[unit]["command"] = mpi_command + command_list[1]
            

    # Convert dictionary to list to write the json file in the proper format.
    output = [output_dict[file] for file in output_dict]
    with open(dir + '/compile_commands_extended.json', 'w') as f:
        print >> f, json.dumps(output, indent=2)
        
        # print 
        # print subprocess.check_output([command, flags])

    # Requires Python 2.7
    
    # for unit in comp_unit_dict:

    
    
