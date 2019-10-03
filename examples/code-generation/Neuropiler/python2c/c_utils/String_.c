#include "utils.h"

Object *new_String(char *base){
    // Intiialize object
    Object *string = new_List();

    // Set default values
    string->name = "string";

    // Set list values
    int i;
    int len = strlen(base);
    for (i = 0; i < len; i++){
        Object *c = new_Char(*(base + i));
        list_append(string, c);
        destroy(c);
    }

    return string;
}