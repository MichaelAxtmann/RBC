cmake_minimum_required(VERSION 3.4)

project(RBC)

set(RBC_VERSION "0.2.0")

# prohibit in-source builds
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif()

# set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

###########################
# cpplint and uncrustify  #
###########################

# test for target to avoid conflicts with parent projects
if (NOT TARGET AnalyzeModifiedSources)
  add_custom_target(AnalyzeModifiedSources  perl "${PROJECT_SOURCE_DIR}/codestyle/analyze-source.pl" "-w")
endif()

if (NOT TARGET AnalyzeAllSources)
add_custom_target(AnalyzeAllSources  perl "${PROJECT_SOURCE_DIR}/codestyle/analyze-source.pl" "-aw")
endif()

########################################################
## External libraries
########################################################

# tlx
add_subdirectory(external/tlx)
set(TLX_LIBRARIES tlx)

########################################################
## Flags
########################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

## enable warnings

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# enable extra warnings on gcc
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
  # disabled to avoid warnings in ips4o
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnoexcept")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
  # enable -Wshadow and -Wold-style-cast only for gcc >= 5
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    # disabled to avoid warnings in ips4o
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast")
  endif()
endif()

# enable extra warnings on clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdeprecated")
endif()

########################################################
## Executables and libraries
########################################################

# rbc library

file(GLOB RBC_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
 src/RBC/Collectives/*.cpp
 src/RBC/PointToPoint/*.cpp
 src/RBC/*.cpp
 )

add_library(rbc STATIC ${RBC_SRCS})
target_link_libraries(rbc tlx)

file(GLOB RBC_INCLUDES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  include/RBC
  external/tlx
  )

target_include_directories(rbc PUBLIC ${RBC_INCLUDES})

# rbc example
add_executable (rbcexample example/rbc_example.cpp)
target_link_libraries(rbcexample rbc)

# test_optimizedcolls binary
enable_testing()

add_executable (test_optimizedcolls test/test_optimizedcolls.cpp)
target_link_libraries(test_optimizedcolls rbc)

find_program (BASH_PROGRAM bash)
if (BASH_PROGRAM)
  add_test(NAME optimizedcolls COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_optimizedcolls.sh ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(optimizedcolls PROPERTIES DEPENDS test_optimizedcolls)
endif()

########################################################
## MPI standard
########################################################

## Check version of MPI standard
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
  "
#include <mpi.h>

int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  int a = 0;
  MPI_Request request;
  MPI_Ibcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD, &request);
  MPI_Wait(&request, MPI_STATUS_IGNORE);
  MPI_Finalize();
}
  "
  NONBLOCKING_COLL_MPI_SUPPORT)

if (NONBLOCKING_COLL_MPI_SUPPORT)
  message(STATUS "MPI supports nonblocking collectives.")
else()
  message(WARNING "MPI does not support nonblocking collectives.")
  target_compile_definitions(rbc PRIVATE NO_NONBLOCKING_COLL_MPI_SUPPORT)
endif()

###########################
# Set model constants     #
###########################

set(CLUSTER_TYPE "UNKNOWN" CACHE STRING "Cluster type")
set_property(CACHE CLUSTER_TYPE PROPERTY STRINGS UNKNOWN SUPERMUC SUPERMUCPHASE2 JUQUEEN)
message(STATUS "Cluster ${CLUSTER_TYPE} selected.")
if ("${CLUSTER_TYPE}" STREQUAL "SUPERMUC")
  target_compile_definitions(rbc PRIVATE "RBC_BETA=3.3e-10")
  target_compile_definitions(rbc PRIVATE "RBC_ALPHA=3.2e-06")
  message(STATUS "Set model parameters: DRBC_BETA=3.3e-10 DRBC_ALPHA=3.2e-06")
elseif ("${CLUSTER_TYPE}" STREQUAL "SUPERMUCPHASE2")
  target_compile_definitions(rbc PRIVATE "RBC_BETA=9.4e-11")
  target_compile_definitions(rbc PRIVATE "RBC_ALPHA=2.5e-06")
  message(STATUS "Set model parameters: DRBC_BETA=9.4e-11 DRBC_ALPHA=2.5e-06")
elseif ("${CLUSTER_TYPE}" STREQUAL "UNKNOWN")
  if (DEFINED RBC_BETA)
    target_compile_definitions(rbc PRIVATE "RBC_BETA=${RBC_BETA}")
    message(STATUS "Set model parameter: RBC_BETA to custom value ${RBC_BETA}")
  else()
    target_compile_definitions(rbc PRIVATE "RBC_BETA=3.3e-10")
    message(STATUS "Set model parameter: RBC_BETA=3.3e-10")
  endif()
  if (DEFINED RBC_ALPHA)
    target_compile_definitions(rbc PRIVATE "RBC_ALPHA=${RBC_ALPHA}")
    message(STATUS "Set model parameter: RBC_ALPHA to custom value ${RBC_ALPHA}")
  else()
    target_compile_definitions(rbc PRIVATE "RBC_ALPHA=3.2e-06")
    message(STATUS "Set model parameter: RBC_ALPHA=3.2e-06")
  endif()
elseif ("${CLUSTER_TYPE}" STREQUAL "JUQUEEN")
  target_compile_definitions(rbc PRIVATE "RBC_BETA=5.63185e-10")
  target_compile_definitions(rbc PRIVATE "RBC_ALPHA=3.54439e-06")
  message(STATUS "Set model parameters: DRBC_BETA=5.63185e-10 DRBC_ALPHA=3.54439e-06")
endif()


###########################
# youcompleteme and rtags #
###########################

option(COMPILE_DATABASE "Creates a compile database for youcompleteme and adds the database to rtags" OFF)

if (COMPILE_DATABASE)
  SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  # add header files to existing compile database
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(add_header_commands_module)
  add_header_commands("${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_CURRENT_BINARY_DIR})

  # compile database created -- continue processing
  IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
    # copy compile database to source directory
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/compile_commands_extended.json
      ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
      )
    MESSAGE(STATUS "Copy file compile_commands_extended.json from binary dir to source dir.")

    # load compile database into rtags
    include(add_database_to_rtags_module)
    add_database_to_rtags(${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
  ENDIF()
endif()
