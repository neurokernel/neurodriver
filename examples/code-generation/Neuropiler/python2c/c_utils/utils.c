#include "utils.h"

/**
 * Convert a static array of chars to a dynamic one
 * by allocating memory. Do this so that there is no
 * concern on whether or not the char array returned
 * by str() should be freed.
 * @param  static_str A static string
 * @return            char*
 */
char *dynamic_str(char *static_str){
	int len = strlen(static_str);
	char *dynamic = (char*)malloc(sizeof(char)*(len+1));
	strncpy(dynamic, static_str, len);
	*(dynamic+len) = 0;
	return dynamic;
}

Object *range(int start, int stop, int step){
	Object *list = new_List();
	int i;
	for (i = start; i < stop; i += step){
		Object *integer = new_Integer(i);
		list_append(list, integer);
		free(integer);
	}
	return list;
}