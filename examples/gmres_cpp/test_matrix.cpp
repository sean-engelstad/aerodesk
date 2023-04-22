#include "core.hpp"
#include "gmres.hpp"
#include <iostream>
#include <complex>
using namespace std;

int main(int argc, char *argv[])
{
    start_random();
    std::cout << "We made it!" << endl;
    Matrix A{2, 2};
    Vector b{2};
    for (int i = 0; i < 2; i++) {
        complex<double> bz{i+1,i-1};
        b.setEntry(i,bz);
        for (int j = 0; j < 2; j++) {
            A.setEntry(i,j,2*i+j);
        }
    }
    // std::cout << "Matrix A ::" << endl;
    // A.printEntries();
    // std::cout << "Vector b ::" << endl;
    // b.printEntries();
    // Matrix C{A*b};
    // std::cout << "Matrix C = A*b ::" << endl;
    // C.printEntries();

    Gmres solver{A,b,100,1.0e-10};
    solver.printMatrices();
}