#ifndef __OBJECT
#define __OBJECT

typedef struct _Object Object;
typedef struct _Node Node;

struct _Node {
	Object *data;
	Node *next;
};

struct _Object {
	// Default values of object.
	// These will always exist for every object.
	char *name;
	int value;

	// List attributes
	unsigned int length;
	Node *head;
	Node *tail;
};

Object *new_Object();
void destroy_Object(Object *obj);
void destroy(Object *obj);

char *str(Object *obj);
char *id(Object *obj);

#endif