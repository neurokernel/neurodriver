#ifndef __INTEGER
#define __INTEGER

Object *new_Integer(int i);
void destroy_Integer(Object *integer);

Object *add_integers(Object *int1, Object *int2);

#endif