#ifndef MATRIX_H
#define MATRIX_H

#include "scalar.hpp"
#include <iostream>

class Matrix
{
public:
    Matrix(int rows_, int cols_) : rows{rows_}, cols{cols_}
    {
        // initialize the matrix storage
        M = new ADScalar[rows_ * cols_];
        // initialize every entry to zero
        for (int i = 0; i < getNumEntries(); i++) {
            M[i] = 0.0;
        }
    }
    int getNumEntries();
    ADScalar getEntry(int irow, int icol);
    ADScalar *getRow(int irow);
    void setEntry(int irow, int icol, ADScalar value);
    void randomInitialize(ADScalar scale = 1.0);
    Matrix operator+(Matrix& right);
    Matrix operator-(Matrix& right);
    Matrix operator*(Matrix& right);
    void printEntries(int cap = 20);

protected:
    int rows,
        cols;
    // pointer to the matrix
    ADScalar *M;
};

#endif // MATRIX_H