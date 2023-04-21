include ../../Makefile.in
include ../../AD_Common.mk

OBJS = test_matrix.o

default: ${OBJS}
		${CXX} -o test_matrix test_matrix.o ${AD_LD_FLAGS}

clean:
		${RM} *.o