#ifndef VECTOR_H
#define VECTOR_H

#include "matrix.hpp"

class Vector : public Matrix {
public:
    Vector(int length) : Matrix(length,1) {};
    ADScalar getEntry(int ind);
    void setEntry(int ind,ADScalar value);
};

#endif // VECTOR_H