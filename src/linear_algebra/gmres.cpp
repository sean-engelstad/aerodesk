#include "gmres.hpp"

void Gmres::printMatrices() {
    std::cout << "A matrix is:" << std::endl;
    A.printEntries();
    std::cout << "b vector is:" << std::endl;
    b.printEntries();
}

void Gmres::setupArnoldi() {
    Q = new Matrix{A.getNumRows(), maxiter+1};
    H = new Matrix{maxiter+1,maxiter};
    xi = new Vector{maxiter+1};
    cosines = new Vector{maxiter};
    sines = new Vector{maxiter};
}

void Gmres::initializeSolution() {
    x = new Vector{b}; // initial guess
    Vector temp{b-A*x};
    //resid = new Vector{temp};
}

void Gmres::jacobiPrecondition() {
    // compute Dinv * A and Dinv * b for each matrix
    // of Jacobi preconditioner
    int ndiag = b.getLength();

    Vector inv_diag{ndiag};
    // read 1/diagonal values
    for (int i = 0; i < ndiag; i++) {
        inv_diag.setEntry(i,1.0/A.getEntry(i,i));
    }
    // multiply Dinv by each A and b
    diagonal_multiply(inv_diag,A);
    diagonal_multiply(inv_diag,b);
}

void Gmres::solve() {
    
}

