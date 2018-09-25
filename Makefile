CXX = mpic++
CXX_FLAGS = -march=native -std=c++17 -O3 -DNDEBUG -g -Wall

.PHONY: rebuild all clean
	
TARGETDIR = build$(SUBDIR)
SRCDIR = .
TESTDIR = ./test
LIBDIR = lib${SUBDIR}
TLXDIR = extlib/tlx
TLXRELEASEDIR = extlib/tlx/build$(SUBDIR)

SRCDIR_RANGE = $(SRCDIR)/RangeBasedComm
TARGETDIR_RANGE = $(TARGETDIR)/RBC
TARGETDIR_TEST = $(TARGETDIR)/test
	
extlib/tlx/Release: 
	mkdir $(TLXRELEASEDIR)

tlx: $(TLXRELEASEDIR)
	$(modules) cd $(TLXRELEASEDIR) && cmake .. -DCMAKE_CXX_COMPILER=$(CXX) && make

all : $(TARGETDIR) $(LIBDIR) $(LIBDIR)/librbc.a $(TARGETDIR)/example $(TARGETDIR_TEST)/test_optimizedcolls
	
# Create libraries
RANGE_OBJ = $(addprefix $(TARGETDIR_RANGE)/,RBC.o Allgather.o Allreduce.o Barrier.o \
	AllreduceTwotree.o ScanTwotree.o ScanAndBcastTwotree.o Twotree.o Bcast.o \
	Exscan.o Gather.o Recv.o Reduce.o Scan.o ScanAndBcast.o Send.o \
	Sendrecv.o) 
$(LIBDIR)/librbc.a : $(TARGETDIR) $(LIBDIR) tlx $(RANGE_OBJ)
	ar rcs $@ $(RANGE_OBJ)
	
# Create the executables	
EXAMPLE = $(LIBDIR)/librbc.a $(TLXRELEASEDIR)/tlx/libtlx.a $(TARGETDIR)/Example.o
$(TARGETDIR)/example : $(EXAMPLE)
	$(CXX) $(CXX_FLAGS) -o $@ $(EXAMPLE) -Llib -lrbc -L$(TLXRELEASEDIR)/tlx/ -ltlx

# Compile Example file
SORT_HEADER = ${addprefix Sort/, Constants.hpp RequestVector.hpp SQuick.hpp \
	    SortingDatatype.hpp TbSplitter.hpp} \
	${addprefix ${SRCDIR}/Sort/SQuick/, DataExchange.hpp PivotSelection.hpp \
	    QSInterval.hpp QuickSort.hpp SequentialSort.hpp}
$(TARGETDIR)/Example.o : $(SRCDIR)/example.cpp $(SORT_HEADER)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<
	
# Create the executables	
TEST_OPTIMIZEDCOLLS = $(LIBDIR)/librbc.a $(TLXRELEASEDIR)/tlx/libtlx.a $(TARGETDIR_TEST)/TestOptimizedcolls.o
$(TARGETDIR_TEST)/test_optimizedcolls : $(TEST_OPTIMIZEDCOLLS)
	$(CXX) $(CXX_FLAGS) -o $@ $(TEST_OPTIMIZEDCOLLS) -Llib -lrbc -L$(TLXRELEASEDIR)/tlx/ -ltlx

test : $(TARGETDIR_TEST)/test_optimizedcolls
	rm -f out_mpi.log
	rm -f out_rbc.log
	touch out_mpi.log
	touch out_rbc.log
	for i in 1 2 3 4 5 6 6 7 8 9 10; do mpirun -np $$i $(TARGETDIR_TEST)/test_optimizedcolls 1; done
	for i in 1 2 3 4 5 6 6 7 8 9 10; do mpirun -np $$i $(TARGETDIR_TEST)/test_optimizedcolls 0; done 
	diff out_mpi.log out_rbc.log

# Compile Test Optimizedcolls file
$(TARGETDIR_TEST)/TestOptimizedcolls.o : $(TESTDIR)/test_optimizedcolls.cpp $(TARGETDIR_TEST) $(LIBDIR)/librbc.a
	$(CXX) $(CXX_FLAGS) -c -o $@ $< -I$(TLXDIR) -I$(SRCDIR_RANGE)

# Compile RangeBasedComm source files	
RANGE_HEADER = $(addprefix $(SRCDIR_RANGE)/,RBC.hpp)
MPI_RANGED_HEADER = $(addprefix $(SRCDIR_RANGE)/,RangeGroup.hpp)

$(TARGETDIR_RANGE)/RBC.o : $(SRCDIR_RANGE)/RBC.cpp $(RANGE_HEADER) $(MPI_RANGED_HEADER)
	$(CXX) $(CXX_FLAGS) -c -o $@ $< -I$(TLXDIR)

$(TARGETDIR_RANGE)/%.o : $(SRCDIR_RANGE)/Collectives/%.cpp $(RANGE_HEADER)
	$(CXX) $(CXX_FLAGS) -c -o $@ $< -I$(TLXDIR)
	
$(TARGETDIR_RANGE)/%.o : $(SRCDIR_RANGE)/PointToPoint/%.cpp $(RANGE_HEADER)
	$(CXX) $(CXX_FLAGS) -c -o $@ $< -I$(TLXDIR)

# Directories
clean : 
	rm -rf $(TARGETDIR)
	rm -rf $(TLXRELEASEDIR)

$(TLXRELEASEDIR) :
	mkdir -p $(TLXRELEASEDIR)

$(TARGETDIR) : $(TARGETDIR_RANGE) $(LIBDIR)
	mkdir -p $(TARGETDIR)
	
$(LIBDIR) :
	mkdir -p $(LIBDIR)
	
$(TARGETDIR_TEST) :
	mkdir -p $(TARGETDIR_TEST)
	
$(TARGETDIR_RANGE) :
	mkdir -p $(TARGETDIR_RANGE)/	
	
rebuild : clean all
