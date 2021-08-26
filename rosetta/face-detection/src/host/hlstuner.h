#include <stdlib.h>

inline void * alloc_input(size_t size)
{
  return malloc(size);
}

inline void free_input(void * ptr)
{
  free(ptr);
}
