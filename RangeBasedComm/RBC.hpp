/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef RBC_RBC_HPP
#define RBC_RBC_HPP

#include <mpi.h>
#include <vector>
#include <memory>
#include <ostream>
#include <functional>

#include "RangeGroup.hpp"

namespace RBC {    

    namespace _internal {

        /**
         * Virtual superclass for the specific requests for each communication operation
         */
        class RequestSuperclass {
        public:

            RequestSuperclass() {
            };
            
            virtual ~RequestSuperclass() {
            };

            //test method has to be implemented by subclasses
            virtual int test(int *flag, MPI_Status *status) = 0;
        };
    }
    
    /**
     * Request class for Range communication.
     * Each non-blocking operation returns a request that is then used to call
     * the Test, Testall, Wait or Waitall operation.
     */
    class Request {
    public:

        Request();
        void set(const std::shared_ptr<_internal::RequestSuperclass>& req);
        Request& operator=(const Request& req);
        int test(int *flag, MPI_Status *status);

    private:
        std::shared_ptr<_internal::RequestSuperclass> req_ptr;
    };
    
    /**
     * Ranged based communicator
     */
    class Comm {
    public:
        
        /**
         * Create an empty communicator.
         * Use RBC::Create_Comm_from_MPI to create a usable communicator.
         */
        Comm();
        
        ~Comm();
        
        /**
         * Only for internal usage
         */
        Comm(MPI_Comm mpi_comm, bool use_mpi_collectives, bool split_mpi_comm, 
                bool use_comm_create);
        /**
         * Only for internal usage
         */
        Comm(RangeGroup range_groups, Comm parent_comm);
        /**
         * Only for internal usage
         */
        Comm(MPI_Comm mpi_comm, Comm parent_comm);
        
        /**
         * Returns the underlying MPI communicator 
         */
        const MPI_Comm& GetMpiComm() const;
        
        /**
         * Returns the rank on the RBC communicator from a rank on the MPI communicator 
         */
        int MpiRankToRangeRank(int mpi_rank) const;
        
        /**
         * Returns the rank on the MPI communicator from a rank on the RBC communicator 
         */
        int RangeRankToMpiRank(int range_rank) const;        
        
        /**
         * Returns true if MPI implementations of collective operations are used whenever possible
         */
        bool useMPICollectives() const;
        
        /**
         * Returns true if the underlying MPI communicator is split in communicator split operations
         */
        bool splitMPIComm() const;        
        
        /**
         * Returns true if the given rank on the MPI communicator is part of the 
         * RBC communicator 
         */
        bool includesMpiRank(int rank) const;
        
        /**
         * Returns true if the communicator contains no ranks 
         */
        bool isEmpty() const;
                        
        void init();
        MPI_Comm mpi_comm;
        bool use_mpi_collectives, split_mpi_comm, use_comm_create;
        bool is_mpi_comm, free_mpi_comm;
        RangeGroup range_group;
        int rank, size;
    };
    std::ostream& operator<<(std::ostream& os, const Comm& comm);

    namespace Model_Const {
        const double
            ALPHA = 1e-6,
            BETA  = 8.55e-11;
    };
    
    namespace Tag_Const {
        const int 
            ALLGATHER                = 1000070,
            ALLREDUCE                = 1000080, 
            ALLREDUCETWOTREE         = 1000081, 
            BARRIER                  = 1000100,
            BCAST                    = 1000105,
            EXSCAN                   = 1000110,
            GATHER                   = 1000115,
            REDUCE                   = 1000120,
            SCAN                     = 1000125,
            SCANTWOTREE              = 1000126,
            SCANANDBCAST             = 1000130,
            SCANANDBCASTSCANTWOTREE  = 1000131,
            SCANANDBCASTBCASTTWOTREE = 1000132;
    };

    namespace Network_Const {
        const double
            TB = 5.63185e-10,
            TS = 3.54439e-06;
    };
    
    /**
     * Non-blocking broadcast
     * @param buffer Buffer where the broadcast value will be stored
     * @param count Number of elements that will be broadcasted
     * @param datatype MPI datatype of the elements
     * @param root The rank that initially has the broadcast value
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
            RBC::Comm const &comm, Request *request, int tag = Tag_Const::BCAST);
    
    /**
     * Blocking broadcast
     * @param buffer Buffer where the broadcast value will be stored
     * @param count Number of elements that will be broadcasted
     * @param datatype MPI datatype of the elements
     * @param root The rank that initially has the broadcast value
     * @param comm The Range comm on which the operation is performed
     */
    int Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
            RBC::Comm const &comm);
    
    /**
     * Non-blocking gather with equal amount of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Number of elements for each receive 
     * @param recvtype MPI datatype of the receive elements
     * @param root Rank of receiving process
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, RBC::Comm const &comm, RBC::Request* request,
            int tag = Tag_Const::GATHER);
    
    /**
     * Blocking gather with equal amount of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Number of elements for each receive 
     * @param recvtype MPI datatype of the receive elements
     * @param root Rank of receiving process
     * @param comm The Range comm on which the operation is performed
     */
    int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, RBC::Comm const &comm);
    
    /**
     * Non-blocking gather with specified number of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcounts Array containing the number of elements that are received from each process 
     * @param displs Array, entry i specifies the displacement relative to recvbuf at which to place the incoming data from process i
     * @param recvtype MPI datatype of the receive elements
     * @param root Rank of receiving process
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype,
            int root, RBC::Comm const &comm, RBC::Request* request,
            int tag = Tag_Const::GATHER);
    
    /**
     * Blocking gather with specified number of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcounts Array containing the number of elements that are received from each process 
     * @param displs Array, entry i specifies the displacement relative to recvbuf at which to place the incoming data from process i
     * @param recvtype MPI datatype of the receive elements
     * @param root Rank of receiving process
     * @param comm The Range comm on which the operation is performed
     */
    int Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype,
            int root, RBC::Comm const &comm);
    
    /**
     * Non-blocking gather that merges the data via a given function
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Total number of all elements that will be received
     * @param root Rank of receiving process
     * @param op Operation that takes (start, mid, end) as parameters and 
     *  merges the two arrays [start, mid) and [mid, end) in-place
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Igatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, int root,
                std::function<void (void*, void*, void*)> op, RBC::Comm const &comm,
                RBC::Request* request, int tag = Tag_Const::GATHER);
    
    /**
     * Blocking gather that merges the data via a given function
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Number of total elements that will be received
     * @param root Rank of receiving process
     * @param op Operation that takes (start, mid, end) as parameters and 
     *  merges the two arrays [start, mid) and [mid, end) in-place
     * @param comm The Range comm on which the operation is performed
     */
    int Gatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, int root,
                std::function<void (void*, void*, void*)> op, RBC::Comm const &comm);
    
    /**
     * Non-blocking allgather with equal amount of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Number of elements for each receive 
     * @param recvtype MPI datatype of the receive elements
     * @param root Rank of receiving process
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Iallgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            RBC::Comm const &comm, RBC::Request* request,
            int tag = Tag_Const::ALLGATHER);
    
    /**
     * Blocking allgather with equal amount of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Number of elements for each receive 
     * @param recvtype MPI datatype of the receive elements
     * @param comm The Range comm on which the operation is performed
     */
    int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            RBC::Comm const &comm);

    namespace _internal {
        namespace optimized {
    
            /**
             * Blocking allgather with equal amount of elements on each process
             * This method uses different implementations depending on the
             * size of comm and the input size.
             * @param sendbuf Starting address of send buffer
             * @param sendcount Number of elements in send buffer
             * @param sendtype MPI datatype of the elements
             * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
             * @param recvcount Number of elements for each receive 
             * @param recvtype MPI datatype of the receive elements
             * @param comm The Range comm on which the operation is performed
             */
            int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    Comm const &comm);
    
            /**
             * Blocking allgather with equal amount of elements on each process
             * This method uses the dissemination algorithm.
             * @param sendbuf Starting address of send buffer
             * @param sendcount Number of elements in send buffer
             * @param sendtype MPI datatype of the elements
             * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
             * @param recvcount Number of elements for each receive 
             * @param recvtype MPI datatype of the receive elements
             * @param comm The Range comm on which the operation is performed
             */
            int AllgatherDissemination(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    Comm const &comm);
    
            /* Allgather operation but any process is allowed to choose its own input size.
             * @param sendbuf Starting address of send buffer
             * @param sendcount Number of elements provided by this process
             * @param sendtype MPI datatype of the elements
             * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
             * @param recvcount Total number of distributed elements.
             * @param recvtype MPI datatype of the receive elements
             * @param comm The Range comm on which the operation is performed
             */
            int AllgathervDissemination(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    Comm const &comm);
    
            /* Allgather-merge operation. Any process is allowed to choose its own input size.
             * @param sendbuf Starting address of send buffer
             * @param sendcount Number of elements provided by this process
             * @param sendtype MPI datatype of the elements
             * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
             * @param recvcount Total number of distributed elements.
             * @param recvtype MPI datatype of the receive elements
             * @param comm The Range comm on which the operation is performed
             * #param merger binary-merge algorithm
             */
            int AllgathervDissemination(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    Comm const &comm,
                    void (*merger)(
                        const void* begin1, const void* end1,
                        const void* begin2, const void* end2,
                        void* out));
    
            /**
             * Blocking allgather with equal amount of elements on each process
             * This method uses the hypercube algorithm.
             * @param sendbuf Starting address of send buffer
             * @param sendcount Number of elements in send buffer
             * @param sendtype MPI datatype of the elements
             * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
             * @param recvcount Number of elements for each receive 
             * @param recvtype MPI datatype of the receive elements
             * @param comm The Range comm on which the operation is performed
             */
            int AllgatherHypercube(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    Comm const &comm);
    
            /**
             * Blocking allgather with equal amount of elements on each process
             * This method uses the pipeline algorithm.
             * @param sendbuf Starting address of send buffer
             * @param sendcount Number of elements in send buffer
             * @param sendtype MPI datatype of the elements
             * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
             * @param recvcount Number of elements for each receive 
             * @param recvtype MPI datatype of the receive elements
             * @param comm The Range comm on which the operation is performed
             */
            int AllgatherPipeline(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    Comm const &comm);

            /**
             * Blocking Allreduce
             * This method uses a hypercube reduce-scatter algorithm
             * followed by a hypercube allgather algorithm. If the
             * number of processes is not a power of two, we do two
             * extra transfers to break down to the next smaller
             * power.
             * @param sendbuf Starting address of send buffer
             * @param recvbuf Starting address of receive buffer
             * @param count Number of elements in send buffer
             * @param datatype MPI datatype of the elements
             * @param op Operation used to reduce two elements
             * @param comm The Range comm on which the operation is performed
             */
            int AllreduceScatterAllgather(const void* sendbuf, void* recvbuf,
                    int count, MPI_Datatype datatype,
                    MPI_Op op, Comm const &comm);

            /**
             * Blocking Allreduce
             * This method uses a hypercube algorithm. If the
             * number of processes is not a power of two, we do two
             * extra transfers to break down to the next smaller
             * power.
             * @param sendbuf Starting address of send buffer
             * @param recvbuf Starting address of receive buffer
             * @param count Number of elements in send buffer
             * @param datatype MPI datatype of the elements
             * @param op Operation used to reduce two elements
             * @param comm The Range comm on which the operation is performed
             */
            int AllreduceHypercube(const void* sendbuf, void* recvbuf,
                    int count, MPI_Datatype datatype,
                    MPI_Op op, Comm const &comm);
    
            /**
             * Blocking Allreduce
             * This method uses different implementations depending on the
             * size of comm and the input size.
             * @param sendbuf Starting address of send buffer
             * @param recvbuf Starting address of receive buffer
             * @param count Number of elements in send buffer
             * @param datatype MPI datatype of the elements
             * @param op Operation used to reduce two elements
             * @param comm The Range comm on which the operation is performed
             */
            int Allreduce(const void* sendbuf, void* recvbuf,
                    int count, MPI_Datatype datatype,
                    MPI_Op op, Comm const &comm);
    
            /**
             * Blocking scan (partial reductions)
             * This method uses a doubling algorithm.
             * @param sendbuf Starting address of send buffer
             * @param recvbuf Starting address of receive buffer
             * @param count Number of elements in send buffer
             * @param datatype MPI datatype of the elements
             * @param op Operation used to reduce two elements
             * @param comm The Range comm on which the operation is performed
             */
            int Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                    MPI_Op op, RBC::Comm const &comm);
    
            /**
             * Blocking exclusive scan (partial reductions)
             * This method uses a doubling algorithm.
             * @param sendbuf Starting address of send buffer
             * @param recvbuf Starting address of receive buffer
             * @param count Number of elements in send buffer
             * @param datatype MPI datatype of the elements
             * @param op Operation used to reduce two elements
             * @param comm The Range comm on which the operation is performed
             */
            int Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                    MPI_Op op, RBC::Comm const &comm);
    
            /**
             * Blocking bcast
             * This method uses a binomial tree algorithm.
             * @param buffer Buffer where the broadcast value will be stored
             * @param count Number of elements that will be broadcasted
             * @param datatype MPI datatype of the elements
             * @param root The rank that initially has the broadcast value
             * @param comm The Range comm on which the operation is performed
             */
            int BcastBinomial(void* buffer, int count, MPI_Datatype datatype, int root,
                    Comm const &comm);

            /**
             * Blocking broadcast
             * This method uses a scatter hypercube-allgather
             * algorithm. If the number of processes is not a power of
             * two, we perform the algorithm on the next smaller power
             * of two and move the results to the remaining processes.
             * @param buffer Buffer where the broadcast value will be stored
             * @param count Number of elements that will be broadcasted
             * @param datatype MPI datatype of the elements
             * @param root The rank that initially has the broadcast value
             * @param comm The Range comm on which the operation is performed
             */
            int BcastScatterAllgather(void* buffer, int count, MPI_Datatype datatype, int root,
                    Comm const &comm);

            /**
             * Blocking broadcast
             * This method uses different implementations depending on the
             * size of comm and the input size.
             * @param buffer Buffer where the broadcast value will be stored
             * @param count Number of elements that will be broadcasted
             * @param datatype MPI datatype of the elements
             * @param root The rank that initially has the broadcast value
             * @param comm The Range comm on which the operation is performed
             */
            int Bcast(void* buffer, int count, MPI_Datatype datatype, int root,
                    Comm const &comm);
            
        } // namespace optimize
    } // end namespace _internal
    
    /**
     * Non-blocking allgather with specified number of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcounts Array containing the number of elements that are received from each process 
     * @param displs Array, entry i specifies the displacement relative to recvbuf at which to place the incoming data from process i
     * @param recvtype MPI datatype of the receive elements
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype,
            RBC::Comm const &comm, RBC::Request* request,
            int tag = Tag_Const::ALLGATHER);
    
    /**
     * Blocking allgather with specified number of elements on each process
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcounts Array containing the number of elements that are received from each process 
     * @param displs Array, entry i specifies the displacement relative to recvbuf at which to place the incoming data from process i
     * @param recvtype MPI datatype of the receive elements
     * @param comm The Range comm on which the operation is performed
     */
    int Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype,
            RBC::Comm const &comm);
    
    /**
     * Non-blocking allgather that merges the data via a given function
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Total number of all elements that will be received
     * @param op Operation that takes (start, mid, end) as parameters and 
     *  merges the two arrays [start, mid) and [mid, end) in-place
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Iallgatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount,
                std::function<void (void*, void*, void*)> op, RBC::Comm const &comm,
                RBC::Request* request, int tag = Tag_Const::ALLGATHER);
    
    /**
     * Blocking allgather that merges the data via a given function
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements in send buffer
     * @param sendtype MPI datatype of the elements
     * @param recvbuf Buffer where the gathered elements will be stored (only relevant at root)
     * @param recvcount Number of total elements that will be received
     * @param op Operation that takes (start, mid, end) as parameters and 
     *  merges the two arrays [start, mid) and [mid, end) in-place
     * @param comm The Range comm on which the operation is performed
     */
    int Allgatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount,
                std::function<void (void*, void*, void*)> op, RBC::Comm const &comm);
    
    
    /**
     * Non-blocking reduce
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param root Rank of receiving process
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Ireduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, RBC::Comm const &comm, Request *request,
            int tag = Tag_Const::REDUCE);
    
    /**
     * Blocking reduce
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param root Rank of receiving process
     * @param comm The Range comm on which the operation is performed
     */
    int Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, RBC::Comm const &comm);

    
    /**
     * Non-blocking Allreduce
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Iallreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, RBC::Comm const &comm, Request *request,
            int tag = Tag_Const::ALLREDUCE);
    
    /**
     * Blocking Allreduce
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     */
    int Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, RBC::Comm const &comm);
    
    /**
     * Blocking Allreduce implemented with the two-tree algorithm
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     */
    int AllreduceTwotree(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, RBC::Comm const &comm);
    
    /**
     * Non-blocking scan (partial reductions)
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Iscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, RBC::Comm const &comm, Request *request,
            int tag = Tag_Const::SCAN);
    
    /**
     * Blocking scan (partial reductions)
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     */
    int Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, RBC::Comm const &comm);
    
    /**
     * Blocking scan (partial reductions) with the two-tree algorithm
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     */
     int ScanTwotree(const void *sendbuf, void *recvbuf, int local_el_cnt, MPI_Datatype datatype,
                     MPI_Op op, RBC::Comm const &comm);

    /**
     * Non-blocking exclusive scan (partial reductions)
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int Iexscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, RBC::Comm const &comm, Request *request,
            int tag = Tag_Const::EXSCAN);
    
    /**
     * Blocking exclusive scan (partial reductions)
     * @param sendbuf Starting address of send buffer
     * @param recvbuf Starting address of receive buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     */
    int Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, RBC::Comm const &comm);
    
    /**
     * Non-blocking scan (partial reductions) and broadcast of the reduction over all elements
     * @param sendbuf Starting address of send buffer
     * @param recvbuf_scan Starting address of receive buffer for the scan value
     * @param recvbuf_bcast Starting address of receive buffer for the broadcast value
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     * @param tag Tag to differentiate between multiple calls
     */
    int IscanAndBcast(const void *sendbuf, void *recvbuf_scan, void *recvbuf_bcast,
            int count, MPI_Datatype datatype, MPI_Op op, RBC::Comm const &comm,
            Request *request, int tag = Tag_Const::SCANANDBCAST);
    
    /**
     * Non-blocking scan (partial reductions) and broadcast of the reduction over all elements
     * @param sendbuf Starting address of send buffer
     * @param recvbuf_scan Starting address of receive buffer for the scan value
     * @param recvbuf_bcast Starting address of receive buffer for the broadcast value
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     */
    int ScanAndBcast(const void *sendbuf, void *recvbuf_scan, void *recvbuf_bcast,
            int count, MPI_Datatype datatype, MPI_Op op, RBC::Comm const &comm);
    
    /**
     * Non-blocking scan (partial reductions) and broadcast of the reduction over all elements
     * implemented with the two-tree algorithm
     * @param sendbuf Starting address of send buffer
     * @param recvbuf_scan Starting address of receive buffer for the scan value
     * @param recvbuf_bcast Starting address of receive buffer for the broadcast value
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param op Operation used to reduce two elements
     * @param comm The Range comm on which the operation is performed
     */
    int ScanAndBcastTwotree(const void *sendbuf, void *recvbuf_scan, void *recvbuf_bcast,
            int count, MPI_Datatype datatype, MPI_Op op, RBC::Comm const &comm);
    
    /**
     * Non-blocking send with MPI_Request
     * @param sendbuf Starting address of send buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param dest Destination rank
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     * @param request MPI_Request that will be returned
     */
    int Isend(const void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, RBC::Comm const &comm, MPI_Request *request);

    /**
     * Non-blocking send
     * @param sendbuf Starting address of send buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param dest Destination rank
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    int Isend(const void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, RBC::Comm const &comm, Request *request);
    
    /**
     * Non-blocking synchronous send
     * @param sendbuf Starting address of send buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param dest Destination rank
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    int Issend(const void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, RBC::Comm const &comm, Request *request);
    
    /**
     * Blocking send
     * @param sendbuf Starting address of send buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param dest Destination rank
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     */
    int Send(const void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, RBC::Comm const &comm);
    
    /**
     * Blocking synchronous send
     * @param sendbuf Starting address of send buffer
     * @param count Number of elements in send buffer
     * @param datatype MPI datatype of the elements
     * @param dest Destination rank
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     */
    int Ssend(const void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, RBC::Comm const &comm);
    
    /**
     * Non-blocking receive with MPI_Request
     * @param sendbuf Starting address of receive buffer
     * @param count Number of elements to be received
     * @param datatype MPI datatype of the elements
     * @param dest Source rank, can be MPI_ANY_SOURCE
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     * @param request MPI_Request that will be returned
     */
    int Irecv(void *buffer, int count, MPI_Datatype datatype, int source,
            int tag, RBC::Comm const &comm, MPI_Request *request);
    
    /**
     * Non-blocking receive
     * @param sendbuf Starting address of receive buffer
     * @param count Number of elements to be received
     * @param datatype MPI datatype of the elements
     * @param dest Source rank, can be MPI_ANY_SOURCE
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    int Irecv(void *buffer, int count, MPI_Datatype datatype, int source,
            int tag, RBC::Comm const &comm, Request *request);
    
    /**
     * Blocking receive
     * @param sendbuf Starting address of receive buffer
     * @param count Number of elements to be received
     * @param datatype MPI datatype of the elements
     * @param dest Source rank, can be MPI_ANY_SOURCE
     * @param tag Tag to differentiate between multiple calls
     * @param comm The Range comm on which the operation is performed
     */
    int Recv(void *buf, int count, MPI_Datatype datatype, int source,
            int tag, RBC::Comm const &comm, MPI_Status *status);

    
    /**
     * Non-blocking send receive operation
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements to be send
     * @param sendtype MPI datatype of the elements
     * @param dest Target rank
     * @param sendtag Tag to differentiate between multiple calls
     * @param recvbuf Starting address of the receive buffer
     * @param recvcount Number of elements to be send
     * @param recvtype MPI datatype of the elements
     * @param source Source rank
     * @param recvtag Tag to differentiate between multiple calls
     * @param comm Communicator
     */
    int Isendrecv(void *sendbuf,
            int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            RBC::Comm const &comm, Request *request);
    
    /**
     * Blocking send receive operation
     * @param sendbuf Starting address of send buffer
     * @param sendcount Number of elements to be send
     * @param sendtype MPI datatype of the elements
     * @param dest Target rank
     * @param sendtag Tag to differentiate between multiple calls
     * @param recvbuf Starting address of the receive buffer
     * @param recvcount Number of elements to be send
     * @param recvtype MPI datatype of the elements
     * @param source Source rank
     * @param recvtag Tag to differentiate between multiple calls
     * @param comm Communicator
     */
    int Sendrecv(void *sendbuf,
            int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            RBC::Comm const &comm, MPI_Status *status);
    
    /**
     * Test if a message can be received
     * @param source Source rank, can be MPI_ANY_SOURCE
     * @param tag Message tag, can be MPI_ANY_TAG
     * @param comm The Range comm on which the operation is performed
     * @param flag Returns 1 if message can be received, else 0
     * @param status Returns a status for the message, can be MPI_STATUS_IGNORE
     * @return 
     */
    int Iprobe(int source, int tag, RBC::Comm const &comm, int *flag, MPI_Status *status);
    
    /**
     * Block until a message can be received
     * @param source Source rank, can be MPI_ANY_SOURCE
     * @param tag Message tag, can be MPI_ANY_TAG
     * @param comm The Range comm on which the operation is performed
     * @param status Returns a status for the message, can be MPI_STATUS_IGNORE
     * @return 
     */
    int Probe(int source, int tag, RBC::Comm const &comm, MPI_Status *status);
    
    /**
     * Non-blocking barrier
     * @param comm The Range comm on which the operation is performed
     * @param request Request that will be returned
     */
    int Ibarrier(RBC::Comm const &comm, Request *request);
    
    /**
     * Blocking barrier
     * @param comm The Range comm on which the operation is performed
     */
    int Barrier(RBC::Comm const &comm);
    
    /**
     * Test if a operation is completed
     * @param request Request of the operation
     * @param flag Returns 1 if operation completed, else 0
     * @param status Returns a status if completed, can be MPI_STATUS_IGNORE
     * @return 
     */
    int Test(Request *request, int *flag, MPI_Status *status);
    
    /**
     * Wait until a operation is completed
     * @param request Request of the operation
     * @param status Returns a status if completed, can be MPI_STATUS_IGNORE
     * @return 
     */
    int Wait(Request *request, MPI_Status *status);
    
    /**
     * Test if multiple operations are completed
     * @param count Number of operations
     * @param array_of_requests Array of requests of the operations
     * @param flag Returns 1 if all operations completed, else 0
     * @param array_of_statuses Array of statuses for the operations, can be MPI_STATUSES_IGNORE
     */
    int Testall(int count, Request array_of_requests[], int *flag,
            MPI_Status array_of_statuses[]);
    
    /**
     * Wait until multiple operations are completed
     * @param count Number of operations
     * @param array_of_requests Array of requests of the operations
     * @param array_of_statuses Array of statuses for the operations, can be MPI_STATUSES_IGNORE
     */
    int Waitall(int count, Request array_of_requests[],
            MPI_Status array_of_statuses[]);
    
    /**
     * Get the rank of this process on the communicator
     * @param comm The Range communicator
     * @param rank Returns the rank
     */
    int Comm_rank(RBC::Comm const &comm, int *rank);
    
    /**
     * Get the size of a Range communicator
     * @param comm The Range communicator
     * @param size Returns the size
     */
    int Comm_size(RBC::Comm const &comm, int *size);

    /**
     * Create a new communicator from a MPI communicator
     * The communicatorr includes all ranks of the MPI communicator
     * @param mpi_comm the MPI communicator
     * @param use_mpi_collectives use native MPI collectives if possible
     * @param split_mpi_comm split the MPI communicator when the RBC communicator is split
     * @param use_comm_create when splitting the MPI communicator, use MPI_Comm_create 
     *  instead of MPI_Comm_split
     * @param comm returns the new communicator
     */
    int Create_Comm_from_MPI(MPI_Comm mpi_comm, RBC::Comm *rcomm,
        bool use_mpi_collectives = false, bool split_mpi_comm = false,
        bool use_comm_create = true);
    
    /**
     * Create a new communicator from a MPI communicator
     * The communicatorr includes all ranks of the MPI communicator
     * @param mpi_comm the MPI communicator
     * @param use_mpi_collectives use native MPI collectives if possible
     * @param split_mpi_comm split the MPI communicator when the RBC communicator is split
     * @param use_comm_create when splitting the MPI communicator, use MPI_Comm_create 
     *  instead of MPI_Comm_split
     * @return the new communicator
     */
    RBC::Comm Create_Comm_from_MPI(MPI_Comm mpi_comm,
        bool use_mpi_collectives = false, bool split_mpi_comm = false,
        bool use_comm_create = true);
    
    /**
     * Create a new communicator that includes a subgroup of ranks [first, last]
     * of the ranks from the old communicator
     * All processes in comm have to call this function
     * @param comm old communicator
     * @param first first rank from the old communicator that is included in the new communicator
     * @param last last rank from the old communicator that is included in the new communicator
     * @param new_comm return the new communicator
     */
    int Comm_create(RBC::Comm const &comm, RBC::Comm *new_comm,
        int first, int last, int stride = 1);
    
    /**
     * Create a new communicator that includes a subgroup of ranks [first, last]
     * of the ranks from the old communicator
     * All processes in comm have to call this function
     * @param comm old communicator
     * @param first first rank from the old communicator that is included in the new communicator
     * @param last last rank from the old communicator that is included in the new communicator
     * @param new_comm return the new communicator
     * @return the new communicator
     */
    RBC::Comm Comm_create(RBC::Comm const &comm,
        int first, int last, int stride = 1);
    
    /**
     * Create a new communicator that includes a subgroup of ranks [first, last]
     * of the ranks from the old communicator
     * Only the processes of the new communicator have to call this function
     * @param comm old communicator
     * @param first first rank from the old communicator that is included in the new communicator
     * @param last last rank from the old communicator that is included in the new communicator
     * @param new_comm return the new communicator
     */
    int Comm_create_group(RBC::Comm const &comm, RBC::Comm *new_comm,
        int first, int last, int stride = 1);

    /**
     * Create two new communicators that include the ranks [first_left, last_left]
     * and [first_right, last_right].
     * first_left <= last_left <= first_right <= last_right
     * @param comm old communicator
     * @param left_start first rank from the old communicator that is included in the new left communicator
     * @param left_end last rank from the old communicator that is included in the new left communicator
     * @param right_start first rank from the old communicator that is included in the new right communicator
     * @param right_end last rank from the old communicator that is included in the new right communicator
     * @param left_comm the new left communicator
     * @param right_comm the new right communicator
     */
    int Split_Comm(RBC::Comm const &comm, int left_start, int left_end, int right_start,
        int right_end, RBC::Comm *left_comm, RBC::Comm *right_comm);
    
    /**
     * Free the MPI communicator if it was created by this library
     * @param comm the Range::Comm including the MPI communicator
     * @param parent_comm this MPI communicator will not be freed
     */
    int Comm_free(RBC::Comm &comm);
    
    /**
     * Returns the rank in the communicator of the source from a MPI_Status
     * @param comm the communicator
     * @param status the status
     */
    int get_Rank_from_Status(RBC::Comm const &comm, MPI_Status status); 
};

#endif /* RBC_RBC_HPP */
