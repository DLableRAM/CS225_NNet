CC=nvcc
CFLAGS=-I

NNet: AItest.o kernelDefs.o
	$(CC) -o NNet AItest.o kernelDefs.o
