#include "matrix.hpp"
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    cout << "We made it!" << endl;
    Matrix mine{2, 2};
    mine.randomInitialize();
    mine.printEntries();
    mine.setEntry(1, 1, 2.0);
    mine.printEntries();
}