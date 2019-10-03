#include "utils.h"

Object *new_Integer(int i){
    // Intiialize object
    Object *integer = new_Object();

    // Set default values
    integer->name = "integer";
    integer->value = i;

    return integer;
}

void destroy_Integer(Object *integer){
    assert(strcmp(integer->name, "integer") == 0);
    free(integer);
}

Object *add_integers(Object *int1, Object *int2){
    int val1 = int1->value;
    int val2 = int2->value;
    return new_Integer(val1 + val2);
}