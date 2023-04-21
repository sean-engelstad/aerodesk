include ../../Makefile.in
include ../../AD_Common.mk

AD_SRC = $(wildcard *.cpp)
AD_OBJ = $(AD_SRC:.cpp=.o)

DIR=${AD_DIR}/src/linear_algebra

default: ${AD_OBJ}
		${AR} ${AR_FLAGS} ${AD_LIB} ${AD_OBJ}

%.o: %.cpp
		${CXX} ${AD_INCLUDE} ${CXXFLAGS} -c $< -o $@

clean:
		${RM} *.o