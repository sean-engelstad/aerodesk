include Makefile.in

AD_SUBDIRS = src/linear_algebra \
	src/linear_algebra/core

AD_OBJS := $(addsuffix /*.o, ${AD_SUBDIRS})

default:
		@for subdir in ${AD_SUBDIRS} ; do \
			echo "making $@ ${subdir}"; \
			echo; (cd $$subdir && $(MAKE) $@ AD_DIR=${AD_DIR}) || exit 1; \
		done
		${CXX} ${SO_LINK_FLAGS} ${AD_OBJS} ${AD_EXTERN_LIBS} -o ${AD_DIR}/lib/libaerodesk.${SO_EXT}

complex:
	@for subdir in ${AD_SUBDIRS} ; do \
		echo "making $@ ${subdir}"; \
		echo; (cd $$subdir && $(MAKE) AD_DIR=${AD_DIR} AD_DEF="${AD_DEF} -DAD_USE_COMPLEX") || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} ${AD_OBJS} ${AD_EXTERN_LIBS} -o ${AD_DIR}/lib/libaerodesk.${SO_EXT}

clean:
	${RM} lib/libaerodesk.a lib/libaerodesk.${SO_EXT}
	@for subdir in ${AD_SUBDIRS} ; do \
		echo "making $@ in ${subdir}"; \
			(cd $$subdir && $(MAKE) $@ AD_DIR=${AD_DIR}) || exit 1; \
	done