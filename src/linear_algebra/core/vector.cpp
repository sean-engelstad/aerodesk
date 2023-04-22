#include "vector.hpp"

ADScalar Vector::getEntry(int ind) {return Matrix::getEntry(ind,0);}
void Vector::setEntry(int ind, ADScalar value) {Matrix::setEntry(ind,0,value);}