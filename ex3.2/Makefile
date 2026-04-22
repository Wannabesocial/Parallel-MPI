CC := mpicc
#CC := gcc

# Flags
CFLAGS := -Wall -fopenmp -Iinc
LDFLAGS := -fopenmp

# Directories
SRC_DIR := src
INC_DIR := inc
BIN_DIR := bin

# Files
SRC := $(wildcard $(SRC_DIR)/*.c)
OBJ := $(patsubst $(SRC_DIR)/%.c,$(BIN_DIR)/%.o,$(SRC)) $(BIN_DIR)/main.o

# Executable
TARGET := ex2


# Default rule
all: $(BIN_DIR) $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) -o $@


# Compile source files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@


$(BIN_DIR)/main.o: main.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@


# Create bin directory if not exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)


# Clean rule
clean:
	rm -rf $(BIN_DIR) $(TARGET)


.PHONY: all compile clean