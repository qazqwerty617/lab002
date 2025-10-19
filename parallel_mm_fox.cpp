// parallel_mm_fox.cpp
// Lab 2: Parallel matrix multiplication using Fox algorithm (C++11 + MPI)
// Build (OpenMPI/MPICH): mpic++ -O2 -std=c++11 parallel_mm_fox.cpp -o parallel_mm_fox
// Run: mpirun -np <p> ./parallel_mm_fox
// Note: Number of processes p must be a perfect square (q*q).
// Matrices are Size x Size, and Size must be divisible by q.

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

int ProcNum = 0;      // Number of processes
int ProcRank = 0;     // Rank of current process
int GridSize = 0;     // q such that ProcNum = q*q

MPI_Comm GridComm;    // 2D Cartesian grid communicator
MPI_Comm RowComm;     // Row communicator
MPI_Comm ColComm;     // Column communicator
int GridCoords[2];    // [row, col] coords of current process in grid

// Utility: print small matrix
void PrintMatrix(const double* M, int rows, int cols, int maxShow=8) {
    int rshow = std::min(rows, maxShow);
    int cshow = std::min(cols, maxShow);
    for (int i = 0; i < rshow; ++i) {
        for (int j = 0; j < cshow; ++j) {
            std::cout << std::setw(8) << std::setprecision(4) << std::fixed
                      << M[i*cols + j] << " ";
        }
        if (cshow < cols) std::cout << "...";
        std::cout << "\n";
    }
    if (rshow < rows) std::cout << "...\n";
}

void DummyDataInitialization(double* A, double* B, int Size) {
    for (int i = 0; i < Size; ++i)
        for (int j = 0; j < Size; ++j) {
            A[i*Size + j] = 1.0;
            B[i*Size + j] = 1.0;
        }
}

void RandomDataInitialization(double* A, double* B, int Size) {
    std::srand(unsigned(std::clock()));
    for (int i = 0; i < Size; ++i)
        for (int j = 0; j < Size; ++j) {
            A[i*Size + j] = (std::rand() % 1000) / 1000.0;
            B[i*Size + j] = (std::rand() % 1000) / 1000.0;
        }
}

// Serial block multiplication: C += A * B for BlockSize x BlockSize blocks
void BlockMultiplication(const double* A, const double* B, double* C, int BlockSize) {
    for (int i = 0; i < BlockSize; ++i)
        for (int k = 0; k < BlockSize; ++k) {
            double aik = A[i*BlockSize + k];
            for (int j = 0; j < BlockSize; ++j) {
                C[i*BlockSize + j] += aik * B[k*BlockSize + j];
            }
        }
}

void CreateGridCommunicators() {
    int dims[2] = { GridSize, GridSize };
    int periods[2] = { 1, 1 };   // both periodic to simplify shifts
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &GridComm);
    MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);

    int subdims_row[2] = { 0, 1 }; // keep columns -> row communicators
    int subdims_col[2] = { 1, 0 }; // keep rows    -> column communicators
    MPI_Cart_sub(GridComm, subdims_row, &RowComm);
    MPI_Cart_sub(GridComm, subdims_col, &ColComm);
}

// Scatter matrix into q x q blocks to all processes
void CheckerboardMatrixScatter(const double* globalM, double* localBlock, int Size, int BlockSize) {
    // Stage 1: scatter stripes (BlockSize rows) down first column (col==0)
    std::vector<double> stripe;
    stripe.resize(BlockSize * Size);
    if (GridCoords[1] == 0) {
        MPI_Scatter(const_cast<double*>(globalM), BlockSize*Size, MPI_DOUBLE,
                    stripe.data(),              BlockSize*Size, MPI_DOUBLE, 0, ColComm);
    }
    // Stage 2: each row scatters blocksize columns to each process in row
    for (int r = 0; r < BlockSize; ++r) {
        const double* sendbuf = (GridCoords[1] == 0) ? stripe.data() + r*Size : nullptr;
        MPI_Scatter(sendbuf, BlockSize, MPI_DOUBLE,
                    localBlock + r*BlockSize, BlockSize, MPI_DOUBLE, 0, RowComm);
    }
}

// Gather q x q blocks from processes into global matrix on root
void CheckerboardMatrixGather(double* globalM, const double* localBlock, int Size, int BlockSize) {
    std::vector<double> stripe;
    stripe.resize(BlockSize * Size);

    // Stage 1: each row gathers from row processes into stripe (on col==0 ranks)
    for (int r = 0; r < BlockSize; ++r) {
        double* recvbuf = (GridCoords[1] == 0) ? stripe.data() + r*Size : nullptr;
        MPI_Gather(localBlock + r*BlockSize, BlockSize, MPI_DOUBLE,
                   recvbuf,                  BlockSize, MPI_DOUBLE, 0, RowComm);
    }
    // Stage 2: first column gathers stripes back to root into globalM
    if (GridCoords[1] == 0) {
        MPI_Gather(stripe.data(), BlockSize*Size, MPI_DOUBLE,
                   globalM,       BlockSize*Size, MPI_DOUBLE, 0, ColComm);
    }
}

// Broadcast needed A block within row for current iter (Fox pivot)
void ABlockCommunication(int iter, double* Ablock, const double* AinitBlock, int BlockSize) {
    int pivot = (GridCoords[0] + iter) % GridSize; // column index within row
    if (GridCoords[1] == pivot) { // copy my initial A block to broadcast buffer
        std::memcpy(Ablock, AinitBlock, sizeof(double)*BlockSize*BlockSize);
    }
    MPI_Bcast(Ablock, BlockSize*BlockSize, MPI_DOUBLE, pivot, RowComm);
}

// Cyclic upward shift of B blocks within a column
void BblockCommunication(double* Bblock, int BlockSize) {
    int up, down;
    MPI_Cart_shift(ColComm, 0, -1, &down, &up); // shift by -1 along column (upward)
    MPI_Status st;
    MPI_Sendrecv_replace(Bblock, BlockSize*BlockSize, MPI_DOUBLE,
                         up, 0, down, 0, ColComm, &st);
}

// Initialize sizes, allocate, and (on root) fill A,B
void ProcessInitialization(double* &A, double* &B, double* &C,
                           double* &Ablock, double* &Bblock, double* &Cblock, double* &AinitBlock,
                           int &Size, int &BlockSize, bool randomize) {
    if (ProcRank == 0) {
        do {
            std::cout << "Enter Size (multiple of " << GridSize << "): ";
            if (!(std::cin >> Size)) { std::cin.clear(); std::cin.ignore(1<<20, '\n'); Size = 0; }
            if (Size <= 0 || (Size % GridSize)!=0)
                std::cout << "Size must be >0 and divisible by grid size.\n";
        } while (Size <= 0 || (Size % GridSize)!=0);
    }
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    BlockSize = Size / GridSize;

    Ablock = new double[BlockSize*BlockSize];
    Bblock = new double[BlockSize*BlockSize];
    Cblock = new double[BlockSize*BlockSize];
    AinitBlock = new double[BlockSize*BlockSize];
    std::fill(Cblock, Cblock + BlockSize*BlockSize, 0.0);

    if (ProcRank == 0) {
        A = new double[Size*Size];
        B = new double[Size*Size];
        C = new double[Size*Size];
        if (randomize) RandomDataInitialization(A, B, Size);
        else DummyDataInitialization(A, B, Size);
    }
}

void ProcessTermination(double* A, double* B, double* C,
                        double* Ablock, double* Bblock, double* Cblock, double* AinitBlock) {
    if (ProcRank == 0) {
        delete [] A; delete [] B; delete [] C;
    }
    delete [] Ablock; delete [] Bblock; delete [] Cblock; delete [] AinitBlock;
}

// Parallel Fox multiplication (C = A * B) using blocks
void ParallelResultCalculation(double* Ablock, double* AinitBlock,
                               double* Bblock, double* Cblock,
                               int BlockSize) {
    for (int iter = 0; iter < GridSize; ++iter) {
        // broadcast the needed A block across the row
        ABlockCommunication(iter, Ablock, AinitBlock, BlockSize);
        // multiply
        BlockMultiplication(Ablock, Bblock, Cblock, BlockSize);
        // shift B upward by one within column
        BblockCommunication(Bblock, BlockSize);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, 0, _IONBF, 0);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    GridSize = (int)std::round(std::sqrt((double)ProcNum));
    if (GridSize * GridSize != ProcNum) {
        if (ProcRank == 0) std::cerr << "Number of processes must be a perfect square.\n";
        MPI_Finalize(); return 1;
    }
    if (ProcRank == 0) std::cout << "Parallel matrix multiplication (Fox) -- q=" << GridSize << "x" << GridSize << "\n";

    CreateGridCommunicators();

    double *A=nullptr, *B=nullptr, *C=nullptr;
    double *Ablock=nullptr, *Bblock=nullptr, *Cblock=nullptr, *AinitBlock=nullptr;
    int Size=0, BlockSize=0;
    bool randomize=false;
    if (ProcRank == 0) { std::cout << "Use random data? (0/1): "; int rnd=0; std::cin >> rnd; randomize = (rnd!=0); }
    MPI_Bcast(&randomize, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    ProcessInitialization(A,B,C, Ablock,Bblock,Cblock,AinitBlock, Size,BlockSize, randomize);

    // Distribute A and B into AinitBlock and Bblock
    CheckerboardMatrixScatter(A, AinitBlock, Size, BlockSize);
    CheckerboardMatrixScatter(B, Bblock,     Size, BlockSize);

    // Optional: print blocks for tiny problems
    if (Size <= 8 && ProcRank==0) {
        std::cout << "Initial A (root):\n"; PrintMatrix(A, Size, Size);
        std::cout << "Initial B (root):\n"; PrintMatrix(B, Size, Size);
    }

    double t0 = MPI_Wtime();
    ParallelResultCalculation(Ablock, AinitBlock, Bblock, Cblock, BlockSize);
    double t1 = MPI_Wtime();

    // Gather C blocks back to root
    CheckerboardMatrixGather(C, Cblock, Size, BlockSize);

    if (ProcRank == 0) {
        if (!randomize && Size <= 10) {
            std::cout << "\nResult C (should be all " << Size << "):\n";
            PrintMatrix(C, Size, Size);
        }
        std::cout << "\nElapsed: " << std::fixed << std::setprecision(6) << (t1 - t0) << " sec\n";
    }

    ProcessTermination(A,B,C, Ablock,Bblock,Cblock,AinitBlock);
    MPI_Finalize();
    return 0;
}
