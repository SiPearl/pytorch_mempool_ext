#define _GNU_SOURCE
#include "numa_allocator.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <linux/mempolicy.h>

/* <linux/mempolicy.h> declares MPOL_PREFERRED_MANY as an enumerator (Linux
 * 5.15 and newer), so this fallback has to come after the include. */
#ifndef MPOL_PREFERRED_MANY
#define MPOL_PREFERRED_MANY 5
#endif

#define BITS_PER_ULONG (sizeof(unsigned long) * CHAR_BIT)
#define NODE_ONLINE_PATH "/sys/devices/system/node/online"
#define NODE_HAS_MEMORY_PATH "/sys/devices/system/node/has_memory"

struct numa_ctx {
  unsigned long* mask;
  /* The kernel reads maxnode - 1 bits from the mask. */
  unsigned long maxnode;
  int mpol_mode;
  int prefault;
  size_t page_size;
};

static void* fail(char* err, size_t errlen, int errnum, const char* fmt, ...) {
  if (err != NULL && errlen > 0) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
  }
  errno = errnum;
  return NULL;
}

/* Returns 1 if node is in the range list at path (for example "0-3,8"), 0 if
 * it is not, and -1 with errno set if the file is unreadable or malformed. */
static int sysfs_list_contains(const char* path, int node) {
  char buf[8192];
  FILE* f = fopen(path, "re");
  if (f == NULL) {
    return -1;
  }
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  int truncated = !feof(f);
  fclose(f);
  if (truncated) {
    errno = EOVERFLOW;
    return -1;
  }
  buf[n] = '\0';

  const char* p = buf;
  while (*p != '\0' && *p != '\n') {
    char* end;
    long lo = strtol(p, &end, 10);
    if (end == p) {
      errno = EINVAL;
      return -1;
    }
    long hi = lo;
    if (*end == '-') {
      p = end + 1;
      hi = strtol(p, &end, 10);
      if (end == p) {
        errno = EINVAL;
        return -1;
      }
    }
    if (node >= lo && node <= hi) {
      return 1;
    }
    if (*end == ',') {
      end++;
    } else if (*end != '\n' && *end != '\0') {
      errno = EINVAL;
      return -1;
    }
    p = end;
  }
  return 0;
}

/* Returns the calling task's allowed-memory mask (the cpuset Mems_allowed) and
 * its size in bits, or NULL with errno set. */
static unsigned long* query_mems_allowed(size_t* nbits_out) {
  /* get_mempolicy returns EINVAL for a mask smaller than the kernel's node
   * count or larger than one page, so grow until it is accepted. */
  for (size_t nbits = 1024; nbits <= 32768; nbits *= 2) {
    unsigned long* mask = calloc(nbits / BITS_PER_ULONG, sizeof(unsigned long));
    if (mask == NULL) {
      return NULL;
    }
    if (syscall(SYS_get_mempolicy, NULL, mask, nbits + 1, NULL, MPOL_F_MEMS_ALLOWED) == 0) {
      *nbits_out = nbits;
      return mask;
    }
    int saved = errno;
    free(mask);
    errno = saved;
    if (saved != EINVAL) {
      return NULL;
    }
  }
  return NULL;
}

static int validate_nodes(const struct numa_alloc_config* cfg, char* err, size_t errlen) {
  size_t allowed_bits = 0;
  unsigned long* allowed = query_mems_allowed(&allowed_bits);
  if (allowed == NULL) {
    int saved = errno;
    fail(err, errlen, saved, "get_mempolicy(MPOL_F_MEMS_ALLOWED) failed: %s", strerror(saved));
    return -1;
  }

  int rc = 0;
  for (size_t i = 0; i < cfg->num_nodes && rc == 0; i++) {
    int node = cfg->nodes[i];
    int online = sysfs_list_contains(NODE_ONLINE_PATH, node);
    int has_memory = online == 1 ? sysfs_list_contains(NODE_HAS_MEMORY_PATH, node) : 0;
    if (online < 0 || has_memory < 0) {
      int saved = errno;
      const char* path = online < 0 ? NODE_ONLINE_PATH : NODE_HAS_MEMORY_PATH;
      fail(err, errlen, saved, "cannot read %s: %s", path, strerror(saved));
      rc = -1;
    } else if (online == 0) {
      fail(err, errlen, EINVAL, "node %d is not online", node);
      rc = -1;
    } else if (has_memory == 0) {
      fail(err, errlen, EINVAL, "node %d has no memory", node);
      rc = -1;
    } else if ((size_t)node >= allowed_bits ||
               !(allowed[(size_t)node / BITS_PER_ULONG] & (1UL << ((size_t)node % BITS_PER_ULONG)))) {
      fail(err, errlen, EINVAL, "node %d is outside Mems_allowed", node);
      rc = -1;
    }
  }
  free(allowed);
  return rc;
}

static int rounded_len(const struct numa_ctx* ctx, size_t nbytes, size_t* len) {
  size_t n = nbytes == 0 ? 1 : nbytes;
  if (n > SIZE_MAX - (ctx->page_size - 1)) {
    return -1;
  }
  *len = (n + ctx->page_size - 1) & ~(ctx->page_size - 1);
  return 0;
}

/* Maps len bytes and applies the policy while no page is populated yet. */
static void* map_bound(const struct numa_ctx* ctx, size_t len) {
  void* p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    return NULL;
  }
  if (syscall(SYS_mbind, p, len, ctx->mpol_mode, ctx->mask, ctx->maxnode, 0U) != 0) {
    int saved = errno;
    munmap(p, len);
    errno = saved;
    return NULL;
  }
  return p;
}

int numa_allocator_abi_version(void) {
  return NUMA_ALLOCATOR_ABI_VERSION;
}

void numa_allocator_destroy(void* handle) {
  struct numa_ctx* ctx = handle;
  if (ctx != NULL) {
    free(ctx->mask);
    free(ctx);
  }
}

void* numa_allocator_create(const struct numa_alloc_config* cfg, char* err, size_t errlen) {
  if (err != NULL && errlen > 0) {
    err[0] = '\0';
  }
  if (cfg == NULL || cfg->nodes == NULL || cfg->num_nodes == 0) {
    return fail(err, errlen, EINVAL, "config needs at least one node");
  }

  int mpol_mode;
  switch (cfg->mode) {
    case NUMA_ALLOC_BIND:
      mpol_mode = MPOL_BIND;
      break;
    case NUMA_ALLOC_INTERLEAVE:
      mpol_mode = MPOL_INTERLEAVE;
      break;
    case NUMA_ALLOC_PREFERRED_MANY:
      mpol_mode = MPOL_PREFERRED_MANY;
      break;
    default:
      return fail(err, errlen, EINVAL, "unknown mode %d", cfg->mode);
  }
  if (cfg->prefault != 0 && cfg->prefault != 1) {
    return fail(err, errlen, EINVAL, "prefault must be 0 or 1, got %d", cfg->prefault);
  }

  int max_node = -1;
  for (size_t i = 0; i < cfg->num_nodes; i++) {
    if (cfg->nodes[i] < 0) {
      return fail(err, errlen, EINVAL, "invalid node id %d", cfg->nodes[i]);
    }
    if (cfg->nodes[i] > max_node) {
      max_node = cfg->nodes[i];
    }
  }
  if (validate_nodes(cfg, err, errlen) != 0) {
    return NULL;
  }

  long page_size = sysconf(_SC_PAGESIZE);
  size_t words = (size_t)max_node / BITS_PER_ULONG + 1;
  struct numa_ctx* ctx = calloc(1, sizeof(*ctx));
  unsigned long* mask = calloc(words, sizeof(unsigned long));
  if (ctx == NULL || mask == NULL || page_size <= 0) {
    free(ctx);
    free(mask);
    return fail(err, errlen, ENOMEM, "out of memory creating the context");
  }
  for (size_t i = 0; i < cfg->num_nodes; i++) {
    size_t node = (size_t)cfg->nodes[i];
    mask[node / BITS_PER_ULONG] |= 1UL << (node % BITS_PER_ULONG);
  }
  ctx->mask = mask;
  ctx->maxnode = words * BITS_PER_ULONG + 1;
  ctx->mpol_mode = mpol_mode;
  ctx->prefault = cfg->prefault;
  ctx->page_size = (size_t)page_size;

  /* Reject a policy the running kernel does not support here instead of at
   * the first allocation. */
  void* probe = map_bound(ctx, ctx->page_size);
  if (probe == NULL) {
    int saved = errno;
    numa_allocator_destroy(ctx);
    return fail(err, errlen, saved, "mbind probe failed: %s", strerror(saved));
  }
  munmap(probe, ctx->page_size);
  return ctx;
}

void* numa_allocator_alloc(void* handle, size_t nbytes) {
  const struct numa_ctx* ctx = handle;
  if (ctx == NULL) {
    errno = EINVAL;
    return NULL;
  }
  size_t len;
  if (rounded_len(ctx, nbytes, &len) != 0) {
    errno = ENOMEM;
    return NULL;
  }
  void* p = map_bound(ctx, len);
  if (p != NULL && ctx->prefault) {
    volatile unsigned char* bytes = p;
    for (size_t off = 0; off < len; off += ctx->page_size) {
      bytes[off] = 0;
    }
  }
  return p;
}

void numa_allocator_free(void* handle, void* ptr, size_t nbytes) {
  const struct numa_ctx* ctx = handle;
  size_t len;
  if (ctx == NULL || ptr == NULL || rounded_len(ctx, nbytes, &len) != 0) {
    return;
  }
  int saved = errno;
  munmap(ptr, len);
  errno = saved;
}
