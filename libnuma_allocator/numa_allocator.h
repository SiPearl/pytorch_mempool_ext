/*
 * libnuma_allocator: page-granular anonymous memory bound to an explicit set of
 * NUMA nodes. C ABI, no dependency on libnuma.
 *
 * Each context owns its node mask and policy. A context is immutable after
 * numa_allocator_create, so numa_allocator_alloc and numa_allocator_free may
 * be called from any thread. The caller must free every allocation before
 * calling numa_allocator_destroy.
 *
 * The numa_allocator_ prefix keeps these names apart from libnuma, which
 * exports numa_alloc and numa_free with different signatures.
 */
#ifndef NUMA_ALLOCATOR_H
#define NUMA_ALLOCATOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUMA_ALLOCATOR_ABI_VERSION 1
#define NUMA_ALLOCATOR_API __attribute__((visibility("default")))

enum numa_alloc_mode {
  NUMA_ALLOC_BIND = 0,           /* MPOL_BIND (default) */
  NUMA_ALLOC_INTERLEAVE = 1,     /* MPOL_INTERLEAVE */
  NUMA_ALLOC_PREFERRED_MANY = 2, /* MPOL_PREFERRED_MANY, Linux 5.15 or newer */
};

struct numa_alloc_config {
  const int* nodes; /* explicit NUMA node IDs, at least one */
  size_t num_nodes;
  int mode;     /* enum numa_alloc_mode */
  int prefault; /* 0: placed on first touch; 1: every page touched at alloc */
};

NUMA_ALLOCATOR_API int numa_allocator_abi_version(void);

/*
 * Returns a context, or NULL on failure with errno set and a message in err.
 * Fails if a node is not online, has no memory, or is outside Mems_allowed,
 * or if the running kernel rejects the policy.
 */
NUMA_ALLOCATOR_API void* numa_allocator_create(
    const struct numa_alloc_config* cfg,
    char* err,
    size_t errlen);

/*
 * Maps nbytes rounded up to whole pages and applies the context's policy
 * before any page is touched. Returns NULL with errno set on failure; there is
 * no fallback to other nodes. nbytes == 0 maps one page, so a non-NULL result
 * always means success.
 *
 * With prefault == 1 and MPOL_BIND, running out of memory on the bound nodes
 * is reported by the kernel as an OOM kill, not as an error return.
 */
NUMA_ALLOCATOR_API void* numa_allocator_alloc(void* ctx, size_t nbytes);

/* nbytes must be the value that was passed to numa_allocator_alloc. ptr may be NULL. */
NUMA_ALLOCATOR_API void numa_allocator_free(void* ctx, void* ptr, size_t nbytes);

NUMA_ALLOCATOR_API void numa_allocator_destroy(void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* NUMA_ALLOCATOR_H */
