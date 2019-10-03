#ifndef __UTILS
#define __UTILS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "Object.h"
#include "Integer.h"
#include "List.h"
#include "Char.h"
#include "String_.h"

char *dynamic_str(char *);

Object *range(int start, int stop, int step);

#endif