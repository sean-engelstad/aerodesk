#ifndef VECTOR_H
#define VECTOR_H

#include "matrix.hpp"

class Vector : public Matrix {
public:
    Vector(int length) : Matrix(length,1) {};
    Vector(Vector& copy) : Matrix(copy) {};
    Vector() : Matrix() {};
    int getLength() {return getNumRows();};
    ADScalar getEntry(int ind);
    void setEntry(int ind,ADScalar value);
};

void diagonal_multiply(Vector& left, Matrix& right) {
    // left*right => into right still
    ADScalar value, diag;
    for (int i = 0; i < left.getNumRows(); i++) {
        diag = left.getEntry(i);
        for (int j = 0; j < right.getNumCols(); j++) {
            value = diag * right.getEntry(i,j);
            right.setEntry(i,j,value);
        }
    }
}

#endif // VECTOR_H