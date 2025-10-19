// serial_mm.cpp
// Lab 2: Serial matrix multiplication (C++11)
// Build (Linux/Mac): g++ -O2 -std=c++11 serial_mm.cpp -o serial_mm
// Build (Windows MSVC): cl /O2 /EHsc serial_mm.cpp
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>
#include <iomanip>

// Pretty print a Size x Size matrix stored row-major in pMatrix
void PrintMatrix(const double* pMatrix, int rows, int cols, int maxShow=10) {
    int rshow = std::min(rows, maxShow);
    int cshow = std::min(cols, maxShow);
    for (int i = 0; i < rshow; ++i) {
        for (int j = 0; j < cshow; ++j) {
            std::cout << std::setw(8) << std::setprecision(4) << std::fixed
                      << pMatrix[i*cols + j] << " ";
        }
        if (cshow < cols) std::cout << "...";
        std::cout << "\n";
    }
    if (rshow < rows) std::cout << "...\n";
}

void DummyDataInitialization(double* pAMatrix, double* pBMatrix, int Size) {
    for (int i = 0; i < Size; ++i) {
        for (int j = 0; j < Size; ++j) {
            pAMatrix[i*Size + j] = 1.0;
            pBMatrix[i*Size + j] = 1.0;
        }
    }
}

void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size) {
    std::srand(unsigned(std::clock()));
    for (int i = 0; i < Size; ++i) {
        for (int j = 0; j < Size; ++j) {
            pAMatrix[i*Size + j] = (std::rand() % 1000) / 1000.0;
            pBMatrix[i*Size + j] = (std::rand() % 1000) / 1000.0;
        }
    }
}

void SerialResultCalculation(const double* A, const double* B, double* C, int Size) {
    for (int i = 0; i < Size; ++i) {
        for (int j = 0; j < Size; ++j) {
            double s = 0.0;
            for (int k = 0; k < Size; ++k) s += A[i*Size + k] * B[k*Size + j];
            C[i*Size + j] = s;
        }
    }
}

void ProcessInitialization(double* &A, double* &B, double* &C, int &Size, bool randomize) {
    do {
        std::cout << "Enter the size of matrices (Size>0): ";
        if (!(std::cin >> Size)) { std::cin.clear(); std::cin.ignore(1<<20, '\n'); Size = -1; }
        if (Size <= 0) std::cout << "Size of objects must be greater than 0!\n";
    } while (Size <= 0);
    A = new double[Size*Size];
    B = new double[Size*Size];
    C = new double[Size*Size];
    if (randomize) RandomDataInitialization(A, B, Size);
    else DummyDataInitialization(A, B, Size);
    std::fill(C, C + Size*Size, 0.0);
}

void ProcessTermination(double* A, double* B, double* C) {
    delete [] A; delete [] B; delete [] C;
}

int main() {
    std::cout << "Serial matrix multiplication program\n";
    double *A=nullptr, *B=nullptr, *C=nullptr;
    int Size = 0;
    bool randomize = false;

    std::cout << "Use random data? (0/1): ";
    int rnd = 0; if (std::cin >> rnd) randomize = (rnd != 0);

    ProcessInitialization(A, B, C, Size, randomize);

    // Optional output for small matrices
    if (Size <= 10) {
        std::cout << "Initial A Matrix\n"; PrintMatrix(A, Size, Size);
        std::cout << "Initial B Matrix\n"; PrintMatrix(B, Size, Size);
    }

    std::clock_t start = std::clock();
    SerialResultCalculation(A, B, C, Size);
    std::clock_t finish = std::clock();
    double duration = (finish - start) / double(CLOCKS_PER_SEC);

    if (Size <= 10) {
        std::cout << "\nResult Matrix:\n"; PrintMatrix(C, Size, Size);
    }
    std::cout << "\nTime of execution: " << std::fixed << std::setprecision(6) << duration << " sec\n";

    ProcessTermination(A, B, C);
    return 0;
}
