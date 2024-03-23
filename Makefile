CC=nvcc
CFLAGS=-c -o
DEPS=defs.cuh
OBJ=neuralnet.o kernelDefs.o interface.o main.o

%.o: %.cu $(DEPS)
	$(CC) $(CFLAGS) $@ $<

NNet: $(OBJ)
	$(CC) $(CFLAGS) $@ $^
