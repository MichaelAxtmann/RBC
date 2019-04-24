function(add_header_commands module_dir dir)
  message("-- Try to find compile commands for included header files...")
  set(code "-cimport sys\; sys.path.insert(0, \"${module_dir}\")\; import add_header_commands\; add_header_commands.add_header_commands(\"${dir}\")")
  # message(${code})
  EXECUTE_PROCESS(COMMAND python ${code} OUTPUT_VARIABLE OUTPUT ERROR_VARIABLE OUTPUT)
  # message(${OUTPUT})
  message("-- Done")
endfunction()
# 

