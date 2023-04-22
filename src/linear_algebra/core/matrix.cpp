
#include "matrix.hpp"

Matrix::Matrix(Matrix& copy) {
    rows = copy.rows;
    cols = copy.cols;
    initialize();
    for (int i = 0; i < getNumEntries(); i++) {
        M[i] = copy.M[i];
    }
}

Matrix::Matrix() {}

int Matrix::getNumEntries()
{
    return rows * cols;
}

ADScalar Matrix::getEntry(int irow, int icol)
{
    return M[cols * irow + icol];
}

void Matrix::setEntry(int irow, int icol, ADScalar value)
{
    M[cols * irow + icol] = value;
}

ADScalar *Matrix::getRow(int irow)
{
    return &M[cols * irow];
}

void Matrix::randomInitialize(ADScalar scale)
{
    for (int i = 0; i < getNumEntries(); i++)
    {
        // choose a random number from 0 to scale (possibly complex)
        M[i] = rand() * ADRealPart(scale) / RAND_MAX;
    }
}

void Matrix::printEntries(int cap)
{
    int irow, icol;
    for (int i = 0; i < getNumEntries() && i < cap; i++)
    {
        irow = i / cols;
        icol = i % cols;
        if (icol == 0)
        {

            std::cout << "Row " << irow << std::endl;
        }
        std::cout << "(" << irow << "," << icol << ") = " << M[i] << std::endl;
    }
}

Matrix Matrix::operator+(Matrix& right) {
    Matrix sum{right.rows,right.cols};
    for (int i = 0; i < right.getNumEntries(); i++) {
        sum.M[i] = this->M[i] + right.M[i];
    }
    return sum;
}

Matrix Matrix::operator-(Matrix& right) {
    Matrix diff{right.rows,right.cols};
    for (int i = 0; i < right.getNumEntries(); i++) {
        diff.M[i] = this->M[i] - right.M[i];
    }
    return diff;
}

Matrix Matrix::operator*(Matrix& right) {
    Matrix product{this->rows, right.cols};
    int rows = this->rows;
    int cols = right.cols;
    int dots = right.rows;
    for (int irow = 0; irow < rows; irow++) {
        for (int icol = 0; icol < cols; icol++) {
            for (int idot = 0; idot < dots; idot++) {
                product.M[cols*irow+icol] += this->M[dots*irow+idot] * right.M[cols*idot + icol];
            }
        }
    }
    return product;
}