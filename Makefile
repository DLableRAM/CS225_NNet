CC=nvcc
CFLAGS= -o
DEPS=defs.cuh
OBJ=AItest.o kernelDefs.o

%.o: %.cu $(DEPS)
	$(CC) -c $(CFLAGS) $@ $<

NNet: $(OBJ)
	$(CC) $(CFLAGS) $@ $^
