# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -I../include

# Source files
SRCS = cubecode/cube.cpp cubecode/glad.c

# Output binary
OUT = cube

# Libraries
LIBS = -lglfw -framework OpenGL

# Targets
all: $(OUT)

$(OUT): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(OUT) $(LIBS)

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT)
