#include <cstdio>
#include <cstdlib>

void error(char* func_name, char* msg) {
    fprintf(stderr, "%s: %s\n", func_name, msg);
    exit(1);
}