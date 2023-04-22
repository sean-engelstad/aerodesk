#ifndef GMRES_H
#define GMRES_H

#include "core/core.hpp"

class Gmres
{
public:
    Gmres(Matrix& A_, Vector &b_, int maxiter_, double tol_) : A{A_}, b{b_},
        maxiter{maxiter_}, tol{tol_} {
            setupArnoldi();
            initializeSolution();
        }
    void setupArnoldi();
    void initializeSolution();
    void printMatrices();
    void jacobiPrecondition();
    void solve();k

private:
    Matrix A, Q, H;
    Vector b, xi, x, cosines, sines, resid, q;
    int maxiter;
    double tol, beta;

};

#endif // GMRES_H