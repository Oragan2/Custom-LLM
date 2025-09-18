CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -O2
NVCCFLAGS = -O2

all: test

test: transformer.o test.o
	$(NVCC) $(NVCCFLAGS) -o test transformer.o test.o

transformer.o: transformer.cu transformer.hpp
	$(NVCC) $(NVCCFLAGS) -c transformer.cu -o transformer.o

test.o: test.cpp transformer.hpp
	$(CXX) $(CXXFLAGS) -c test.cpp -o test.o

clean:
	rm -f *.o test

doc:
	doxygen ../../Doxyfile
