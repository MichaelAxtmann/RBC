CXX = mpic++ -march=native -std=c++11 -O3 -DNDEBUG -g -Wall

.PHONY: rebuild all clean
	
TARGETDIR = build$(SUBDIR)
SRCDIR = .
LIBDIR = lib${SUBDIR}

SRCDIR_RANGE = $(SRCDIR)/RangeBasedComm
TARGETDIR_RANGE = $(TARGETDIR)/RBC
	
all : $(TARGETDIR) $(LIBDIR) $(LIBDIR)/librbc.a $(TARGETDIR)/example
	
# Create the executables	
EXAMPLE = $(TARGETDIR)/Example.o $(LIBDIR)/librbc.a
$(TARGETDIR)/example : $(EXAMPLE)
	$(CXX) -o $@ $(EXAMPLE)
	
# Create libraries
RANGE_OBJ = $(addprefix $(TARGETDIR_RANGE)/,RBC.o Allgather.o Allreduce.o Barrier.o \
	AllreduceTwotree.o ScanTwotree.o ScanAndBcastTwotree.o Twotree.o Bcast.o \
	Exscan.o Gather.o Recv.o Reduce.o Scan.o ScanAndBcast.o Send.o \
	Sendrecv.o) 
$(LIBDIR)/librbc.a : $(RANGE_OBJ)
	ar rcs $@ $(RANGE_OBJ)

# Compile Example file
SORT_HEADER = ${addprefix Sort/, Constants.hpp RequestVector.hpp SQuick.hpp \
	    SortingDatatype.hpp TbSplitter.hpp} \
	${addprefix ${SRCDIR}/Sort/SQuick/, DataExchange.hpp PivotSelection.hpp \
	    QSInterval.hpp QuickSort.hpp SequentialSort.hpp}
$(TARGETDIR)/Example.o : $(SRCDIR)/example.cpp $(SORT_HEADER)
	$(CXX) -c -o $@ $<
	
# Compile RangeBasedComm source files	
RANGE_HEADER = $(addprefix $(SRCDIR_RANGE)/,RBC.hpp)
MPI_RANGED_HEADER = $(addprefix $(SRCDIR_RANGE)/,RangeGroup.hpp)

$(TARGETDIR_RANGE)/RBC.o : $(SRCDIR_RANGE)/RBC.cpp $(RANGE_HEADER) $(MPI_RANGED_HEADER)
	$(CXX) -c -o $@ $<

$(TARGETDIR_RANGE)/%.o : $(SRCDIR_RANGE)/Collectives/%.cpp $(RANGE_HEADER)
	$(CXX) -c -o $@ $<
	
$(TARGETDIR_RANGE)/%.o : $(SRCDIR_RANGE)/PointToPoint/%.cpp $(RANGE_HEADER)
	$(CXX) -c -o $@ $<

# Directories
clean : 
	rm -rf $(TARGETDIR)

$(TARGETDIR) : $(TARGETDIR_RANGE) $(LIBDIR)
	mkdir -p $(TARGETDIR)
	
$(LIBDIR) :
	mkdir -p $(LIBDIR)
	
$(TARGETDIR_RANGE) :
	mkdir -p $(TARGETDIR_RANGE)/	
	
rebuild : clean all
