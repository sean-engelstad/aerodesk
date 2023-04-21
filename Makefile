include Makefile.in

AD_SUBDIRS = src/linear_algebra \
	src/linear_algebra/utils

AD_OBJS := $(addsuffix /*.o, ${AD_SUBDIRS})

default:
		for subdir in ${AD_SUBDIRS} ; do \
			echo "making ${subdir}"; \
			echo; (cd $$subdir && $(MAKE) AD_DIR=${AD_DIR}) || exit 1; \
		done
		${CXX} ${SO_LINK_FLAGS} ${AD_OBJS} ${AD_EXTERN_LIBS} -o ${AD_DIR}/lib/libaerodesk.${SO_EXT}

clean:
	for subdir in ${AD_SUBDIRS} ; do \
		echo "removing in ${subdir}"; \
	done