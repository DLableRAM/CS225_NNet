CC=nvcc
CFLAGS= -G -o
DEPS=defs.cuh
OBJ=neuralnet.o kernelDefs.o interface.o main.o

%.o: %.cu $(DEPS)
	$(CC) -c $(CFLAGS) $@ $<

NNet: $(OBJ)
	$(CC) $(CFLAGS) $@ $^
