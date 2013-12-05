

CC=mpicc
CCFLAGS+= -O2 -I${LEVEL}/utility

UTIL_DIR= ${LEVEL}/utility

OBJS = ${UTIL_DIR}/utility.o ${EXAMPLE}.o


P=16
RUN=mpiexec

default:${EXAMPLE}

${UTIL_DIR}/utility.o : ${UTIL_DIR}/utility.h
	pushd ${UTIL_DIR}; ${CC} -c ${CCFLAGS} utility.c; popd
	

${EXAMPLE}.o : ${EXAMPLE}.c
	${CC} -c ${CCFLAGS} $<

${EXAMPLE}: ${OBJS}
	${CC} ${OBJS} ${CCFLAGS} -o $@


run: ${EXAMPLE}
	${RUN} -n ${P} ./$< ${RUN_ARGS}

qrun: ${EXAMPLE} ${EXAMPLE}.pbs
	qsub ${EXAMPLE}.pbs
	
clean:
	rm -f ${EXAMPLE} ${OBJS}
