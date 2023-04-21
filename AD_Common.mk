AD_LIB = ${AD_DIR}/lib/libaerodesk.a

AD_INCLUDE = -I${AD_DIR}/src/linear_algebra \
	-I${AD_DIR}/src/linear_algebra/utils

AD_CC_FLAGS = ${AD_INCLUDE}

%.o: %.cpp
	${CXX} ${AD_CC_FLAGS} -c $< -o $*.o
	@echo
	@echo "        --- Compiled $*.cpp successfully ---"
	@echo