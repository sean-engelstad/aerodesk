#ifndef CORE_H
#define CORE_H

#include <iostream>
#include <time.h>

void start_random() {
    srand((unsigned)time(NULL)); // random seed
}

// file for including all matrix-vector core operations
#include "matrix.hpp"
#include "vector.hpp"
#endif // CORE_H