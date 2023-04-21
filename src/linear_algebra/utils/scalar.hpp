#ifndef SCALAR_H
#define SCALAR_H

#include <complex>
#include <cmath>

typedef double ADReal;
typedef std::complex<double> ADComplex;

// define the switch for complex versus real scalars
#ifdef AERODESK_USE_COMPLEX
typedef ADComplex ADScalar;
#else
typedef ADReal ADScalar;
#endif

// define basic complex-number operations on ADScalar objects for each subtype
inline double ADRealPart(const ADComplex &c) { return real(c); }
inline double ADImagPart(const ADComplex &c) { return imag(c); }
inline double ADRealPart(const ADReal &d) { return d; }
inline double ADImagPart(const ADReal &d) { return 0.0; }
inline double ADabs(const ADReal &d) { return abs(d); }
inline double ADabs(const ADComplex &c)
{
    return std::pow(ADRealPart(c) * ADRealPart(c) + ADImagPart(c) * ADImagPart(c), 0.5);
}
#endif // SCALAR_H