#ifndef _NVDS_ALLOCATOR_H_
#define _NVDS_ALLOCATOR_H_

#include <stdint.h>

class INvDsAllocator {
public:
    /**
     * @brief  Allocate memory of @size Bytes
     * @param  size [IN] The # of bytes to allocate
     * @return The newly allocated Memory buffer handle
     */
    virtual void *Allocate(uint32_t size) = 0;
    /**
     * @brief Deallocate the memory allocated using Allocate()
     * @param data [IN] The Memory buffer handle
     */
    virtual void Deallocate(void *data) = 0;
    virtual ~INvDsAllocator() {}
};

#endif
