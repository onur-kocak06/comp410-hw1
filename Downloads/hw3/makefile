# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -I../include

# Source files
SRCS = bounce/main.cpp bounce/glad.c

# Output binary
OUT = bouncing_ball

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
