#ifndef _NVDS_MEMORY_ALLOCATOR_H_
#define _NVDS_MEMORY_ALLOCATOR_H_

#include "INvDsAllocator.h"

/**
 * Specifies memory types for \ref NvDsMemory.
 */
typedef enum {
    NVDS_MEM_DEFAULT,
    /** Specifies CUDA Host memory type. */
    NVDS_MEM_CUDA_PINNED,
    /** Specifies CUDA Device memory type. */
    NVDS_MEM_CUDA_DEVICE,
    /** Specifies CUDA Unified memory type. */
    NVDS_MEM_CUDA_UNIFIED,
    /** Specifies memory allocated by malloc(). */
    NVDS_MEM_SYSTEM,
} NvDsMemType;

class NvDsMemoryAllocator : INvDsAllocator {
public:
    NvDsMemoryAllocator(uint32_t gpuId, NvDsMemType memType);
    void *Allocate(uint32_t size);
    void Deallocate(void *data);

private:
    uint32_t gpuId;
    NvDsMemType memType;
};

#endif
