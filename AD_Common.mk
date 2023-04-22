AD_LIB = ${AD_DIR}/lib/libaerodesk.a

AD_INCLUDE = -I${AD_DIR}/src/linear_algebra \
	-I${AD_DIR}/src/linear_algebra/core

AD_CC_FLAGS = ${AD_DEF} ${AD_INCLUDE} ${CXXFLAGS}

%.o: %.cpp
	${CXX} ${AD_CC_FLAGS} -c $< -o $*.o
	@echo
	@echo "        --- Compiled $*.cpp successfully ---"
	@echo