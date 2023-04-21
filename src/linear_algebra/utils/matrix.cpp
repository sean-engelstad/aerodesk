
#include "matrix.hpp"

int Matrix::getNumEntries()
{
    return rows * cols;
}

ADScalar Matrix::getEntry(int irow, int icol)
{
    return M[rows * irow + icol];
}

void Matrix::setEntry(int irow, int icol, ADScalar value)
{
    M[rows * irow + icol] = value;
}

ADScalar *Matrix::getRow(int irow)
{
    return &M[rows * irow];
}

void Matrix::randomInitialize(ADScalar scale)
{
    srand((unsigned)time(NULL)); // random seed
    for (int i = 0; i < getNumEntries(); i++)
    {
        // choose a random number from 0 to scale (possibly complex)
        M[i] = rand() * scale / RAND_MAX;
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