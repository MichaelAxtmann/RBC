#!/bin/bash

TARGETDIR_TEST=$1

for i in {1..11}; do
    for EXP in {1..14}; do
        rm -f out_mpi.log
        rm -f out_rbc.log
        touch out_mpi.log
        touch out_rbc.log
        echo "Run benchmark on $i PEs with $((2**${EXP})) $((2**${EXP}+1)) and $((2**${EXP}-1)) elements." 
        mpirun -np $i ${TARGETDIR_TEST}/test_optimizedcolls 0 $((2**${EXP}))
        mpirun -np $i ${TARGETDIR_TEST}/test_optimizedcolls 1 $((2**${EXP}))
        DIFF=$(diff out_mpi.log out_rbc.log)
        if [ "${DIFF}" ]
        then
            echo "$DIFF" > test.log
            exit 1
        fi
        mpirun -np $i ${TARGETDIR_TEST}/test_optimizedcolls 0 $((2**${EXP}-1))
        mpirun -np $i ${TARGETDIR_TEST}/test_optimizedcolls 1 $((2**${EXP}-1))
        DIFF=$(diff out_mpi.log out_rbc.log)
        if [ "${DIFF}" ]
        then
            echo "$DIFF" > test.log
            exit 1
        fi
        mpirun -np $i ${TARGETDIR_TEST}/test_optimizedcolls 0 $((2**${EXP}+1))
        mpirun -np $i ${TARGETDIR_TEST}/test_optimizedcolls 1 $((2**${EXP}+1))
        DIFF=$(diff out_mpi.log out_rbc.log)
        if [ "${DIFF}" ]
        then
            echo "$DIFF" > test.log
            exit 1
        fi
    done
done
for i in {1..20}; do
    for NUM_ELS in {1..100}; do
        rm -f out_mpi.log
        rm -f out_rbc.log
        touch out_mpi.log
        touch out_rbc.log
        echo "Run benchmark on $i PEs with ${NUM_ELS} elements."
        mpirun -np $i ${TARGETDIR_TEST}/test_optimizedcolls 0 ${NUM_ELS}
        mpirun -np $i ${TARGETDIR_TEST}/test_optimizedcolls 1 ${NUM_ELS}
        DIFF=$(diff out_mpi.log out_rbc.log)
        if [ "${DIFF}" ]
        then
            echo "$DIFF" > test.log
            exit 1
        fi
    done
done
