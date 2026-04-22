#ifndef VALID_H
#define VALID_H

#include <stdbool.h>
#include "structs.h"

bool valid_vectors(const _vector *v1, const _vector *v2);

void valid_arguments(const int argc, const char **argv);

#endif