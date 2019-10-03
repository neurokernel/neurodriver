#include "utils.h"

Object *new_Char(char c){
    // Intiialize object
    Object *char_ = new_Object();

    // Set default values
    char_->name = "char";
    char_->value = c;

    return char_;
}

void destroy_Char(Object *c){
    assert(strcmp(c->name, "char") == 0);
    free(c);
}