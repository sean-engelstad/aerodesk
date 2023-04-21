#ifndef MATRIX_H
#define MATRIX_H

#include "scalar.hpp"
#include <iostream>
#include <time.h>

class Matrix
{
public:
    Matrix(int rows_, int cols_) : rows{rows_}, cols{cols_}
    {
        // initialize the matrix storage
        M = new ADScalar[rows_ * cols_];
    }
    int getNumEntries();
    ADScalar getEntry(int, int);
    ADScalar *getRow(int);
    void setEntry(int irow, int icol, ADScalar value);
    void randomInitialize(ADScalar scale = 1.0);
    Matrix operator*(Matrix right);
    Matrix operator+(Matrix right);
    void printEntries(int cap = 20);

private:
    int rows,
        cols;
    // pointer to the matrix
    ADScalar *M;
};

#endif // MATRIX_H