#include "utils.h"

Object *new_List(){
    // Intiialize object
    Object *list = new_Object();

    // Set default values
    list->name = "list";

    // Set list values
    list->length = 0;
    list->head = list->tail = NULL;

    return list;
}

void destroy_List(Object *list){
    assert(strcmp(list->name, "list") == 0 ||
           strcmp(list->name, "string") == 0);

    // Free each element in the list and the data it holds.
    Node *current;
    while(list->head != NULL) {
        current = list->head;
        list->head = current->next;
        free(current->data); // Use free (not destroy) because data was malloc'd
        free(current);
    }

    free(list);
}

/**
 * Copy an element and prepend it to the front of
 * the list. Remeber to free this element and the
 * data in it later.
 * @param list List to get a new elem
 * @param elem Elem to get prepended
 */
void list_prepend(Object *list, Object *elem){
    Node *node = (Node*)malloc(sizeof(Node));
    node->data = (Object*)malloc(sizeof(Object));
    memcpy(node->data, elem, sizeof(Object));

    // Set the old head as the new node next.
    // Set the new node as the new list head.
    node->next = list->head;
    list->head = node;

    // If the element is first elem in the list,
    // the head could still point to null, so
    // reassign the head to itself.
    if(!list->tail) {
        list->tail = list->head;
    }

    list->length++;
}

/**
 * Copy an element and append it to the front of
 * the list. Remeber to free this element and the
 * data in it later.
 * @param list List to get a new elem
 * @param elem Elem to get appended
 */
void list_append(Object *list, Object *elem){
    Node *node = (Node*)malloc(sizeof(Node));
    node->data = (Object*)malloc(sizeof(Object));
    memcpy(node->data, elem, sizeof(Object));

    // Make the new node next null and the new tail.
    // If the list was initially empty, make the new
    // node the new head also.
    node->next = NULL;
    if(list->length == 0) {
        list->head = list->tail = node;
    } else {
        list->tail->next = node;
        list->tail = node;
    }

    list->length++;
}

/**
 * Get the ith element of the list object.
 * @param  list List to loop through
 * @param  i    Index
 * @return      Pointer to object (not copy)
 */
Object *list_get(Object *list, unsigned int i){
    assert(i < list->length);

    Node *current = list->head;
    int j = 0;
    for (j = 0; j < i; j++){
        current = current->next;
    }
    return current->data;
}


char *list_str(Object *list){
    // Return the contents of the list separated by ,
    char *str_rep = (char*)malloc(sizeof(char)*3);
    unsigned int len = 3; // Initially just "[]" (then null terminator)
    unsigned int start = 1; // Start after the [
    *str_rep = '[';

    int i;
    Object *elem = NULL;
    for (i = 0; i < list->length; i++){
        // I know this is an inefficient way of getting the elems
        // but I am just trying to get this to work for now.
        elem = list_get(list, i);
        char *elem_str = str(elem);
        int elem_len = strlen(elem_str);

        if (strcmp(elem->name, "string") == 0){
            // Resize the list str_rep
            len += elem_len + 4; // The string len + ', ' + 2 double quotes
            str_rep = (char*)realloc(str_rep, sizeof(char)*len);
            *(str_rep + start) = '\'';
            strncpy(str_rep + start + 1, elem_str, elem_len);
            *(str_rep + start + 1 + elem_len) = '\'';
            *(str_rep + start + 1 + elem_len+1) = ',';
            *(str_rep + start + 1 + elem_len+2) = ' ';
            start += elem_len + 4;
        }
        else {
            // Resize the list str_rep
            len += elem_len + 2; // The string len + ', '
            str_rep = (char*)realloc(str_rep, sizeof(char)*len);
            strncpy(str_rep + start, elem_str, elem_len);
            *(str_rep + start + elem_len) = ',';
            *(str_rep + start + elem_len+1) = ' ';
            start += elem_len + 2;
        }

        free(elem_str);
    }

    if (list->length > 0){
        len -= 2;
    }

    *(str_rep+len-2) = ']';
    *(str_rep+len-1) = 0;
    return str_rep;
}

