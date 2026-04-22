#include "valid.h"

bool valid_vectors(const _vector *v1, const _vector *v2){

    if(v1->size != v2->size) return false;

    for(int i = 0; i < v1->size; i++){
        if(v1->array[i] != v2->array[i])
            return false;
    }

    return true;
}

void valid_arguments(const int argc, const char **argv){

    int size, sparsity, mult;

    // Problem: Not enough arguments
    if(argc != 4){
        HANDLE_ERROR("Usage: ./<executable> <array size> <sparsity> <mult times>");
    }

    size = atoi(argv[1]), sparsity = atoi(argv[2]), mult = atoi(argv[3]);

    // Problem: Out of range <array size>
    if(size <= 0 || size > 10000){
        HANDLE_ERROR("Out of range <array size>. Range [1, 10^4]");
    }

    // Problem: Out of range <sparsity>
    if(sparsity < 0 || sparsity > 99){
        HANDLE_ERROR("Out of range <sparsity>. Range [0%, 99%]");
    }

    // Problem: Out of range <mult times>
    if(mult < 1 || mult > 20){
        HANDLE_ERROR("Out of range <mult times>. Range [1, 20]");
    }
}


