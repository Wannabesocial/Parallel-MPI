SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

EXEC_NAMES := polynomial

EXE := $(addprefix $(BIN_DIR)/, $(EXEC_NAMES))
ALL_OBJ := $(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(EXEC_NAMES)))

POLYNOMIAL_DEPS := $(OBJ_DIR)/polynomial-funcs.o


CC := mpicc
CFLAGS := -Wall -Iinclude -MMD -MP $(EXTRA_CFLAGS)
LDFLAGS :=

.PHONY: all clean $(EXEC_NAMES)

all: $(EXE)

$(EXEC_NAMES): %: $(BIN_DIR)/%


$(BIN_DIR)/polynomial: $(POLYNOMIAL_DEPS)

$(BIN_DIR)/%: $(OBJ_DIR)/%.o | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

-include $(OBJ_DIR)/*.d
