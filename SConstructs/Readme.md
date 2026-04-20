# SConstruct Workflows

Files are organized in nested zipped folders. Each example must be run individually before figures can be generated in the main folder.

## Running the Examples

1. Unzip the folder for the example you want to run.
2. Ensure all supplemental executables have been compiled and that the paths in the SConstruct file point to them correctly.
3. Run the example using Madagascar's `scons` command:
   - **Sequential:** `scons`
   - **Parallel (across nodes):** Submit via SLURM to a GPU node 
     - Use the included `mycscons` script as `./mycscons -f SConstruct_filename` 
     - This submits jobs to a GPU node using Madagascar's cluster module
     - Run `./mycscons -c` to clean up job files.

All SConstructs have been configured to support both sequential and parallelized execution.

## Additional Resources

For more information on the SConstruct file structure and how to run Madagascar workflows, see:
https://ahay.org/wiki/Revisiting_SEP_tour_with_Madagascar_and_SCons