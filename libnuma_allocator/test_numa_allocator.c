/*
 * Smoke test for libnuma_allocator.
 *
 *   test_numa_allocator [NODES_A [NODES_B]]     (defaults: "0" and "1")
 *
 * NODES_* are lists such as "0-3" or "4,6". Instance A must be usable on this
 * machine. If NODES_B is not, the checks that need a second node set are
 * skipped and instance B falls back to NODES_A with MPOL_INTERLEAVE, so two
 * independent instances are still exercised.
 */
#define _GNU_SOURCE
#include "numa_allocator.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <linux/mempolicy.h>

#ifndef MPOL_PREFERRED_MANY
#define MPOL_PREFERRED_MANY 5
#endif

#define MAX_NODES 1024
#define MASK_WORDS (MAX_NODES / (sizeof(unsigned long) * CHAR_BIT))
#define BIG_BYTES ((size_t)32 << 20)

static int failures;
static int skips;

#define CHECK(cond, ...)                                    \
  do {                                                      \
    if (!(cond)) {                                          \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);  \
      fprintf(stderr, __VA_ARGS__);                         \
      fputc('\n', stderr);                                  \
      failures++;                                           \
    }                                                       \
  } while (0)

struct node_set {
  int nodes[MAX_NODES];
  size_t count;
};

static int parse_nodes(const char* text, struct node_set* out) {
  out->count = 0;
  const char* p = text;
  while (*p != '\0') {
    char* end;
    long lo = strtol(p, &end, 10);
    if (end == p || lo < 0) {
      return -1;
    }
    long hi = lo;
    if (*end == '-') {
      p = end + 1;
      hi = strtol(p, &end, 10);
      if (end == p || hi < lo) {
        return -1;
      }
    }
    for (long n = lo; n <= hi; n++) {
      if (n >= MAX_NODES || out->count == MAX_NODES) {
        return -1;
      }
      out->nodes[out->count++] = (int)n;
    }
    if (*end == ',') {
      end++;
    } else if (*end != '\0') {
      return -1;
    }
    p = end;
  }
  return out->count > 0 ? 0 : -1;
}

static int in_set(const struct node_set* set, int node) {
  for (size_t i = 0; i < set->count; i++) {
    if (set->nodes[i] == node) {
      return 1;
    }
  }
  return 0;
}

static void* create(const struct node_set* set, int mode, int prefault, char* err, size_t errlen) {
  struct numa_alloc_config cfg = {
      .nodes = set->nodes, .num_nodes = set->count, .mode = mode, .prefault = prefault};
  return numa_allocator_create(&cfg, err, errlen);
}

static int mpol_of(int mode) {
  switch (mode) {
    case NUMA_ALLOC_INTERLEAVE:
      return MPOL_INTERLEAVE;
    case NUMA_ALLOC_PREFERRED_MANY:
      return MPOL_PREFERRED_MANY;
    default:
      return MPOL_BIND;
  }
}

/* Fills status with the node of each page, or a negative errno (-ENOENT for a
 * page that is not populated). */
static int residency(void* ptr, size_t pages, size_t page_size, int* status) {
  void** addrs = malloc(pages * sizeof(void*));
  if (addrs == NULL) {
    return -1;
  }
  for (size_t i = 0; i < pages; i++) {
    addrs[i] = (char*)ptr + i * page_size;
  }
  long rc = syscall(SYS_move_pages, 0, pages, addrs, NULL, status, 0);
  free(addrs);
  return rc == 0 ? 0 : -1;
}

static void check_vma_policy(void* ptr, const struct node_set* set, int mode, const char* label) {
  unsigned long mask[MASK_WORDS] = {0};
  int got_mode = -1;
  if (syscall(SYS_get_mempolicy, &got_mode, mask, MAX_NODES + 1, ptr, MPOL_F_ADDR) != 0) {
    CHECK(0, "%s: get_mempolicy(MPOL_F_ADDR) failed: %s", label, strerror(errno));
    return;
  }
  CHECK(got_mode == mpol_of(mode), "%s: VMA policy mode %d, expected %d", label, got_mode, mpol_of(mode));
  for (int node = 0; node < MAX_NODES; node++) {
    int bit = (mask[(size_t)node / (sizeof(unsigned long) * CHAR_BIT)] >>
               ((size_t)node % (sizeof(unsigned long) * CHAR_BIT))) & 1;
    CHECK(bit == in_set(set, node), "%s: VMA policy mask bit %d is %d", label, node, bit);
  }
}

static void exercise(void* ctx, const struct node_set* set, int mode, int prefault, const char* label) {
  size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
  size_t sizes[] = {0, 1, page_size, 3 * page_size + 1, BIG_BYTES};

  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
    size_t nbytes = sizes[s];
    size_t pages = nbytes == 0 ? 1 : (nbytes + page_size - 1) / page_size;
    unsigned char* p = numa_allocator_alloc(ctx, nbytes);
    CHECK(p != NULL, "%s: numa_allocator_alloc(%zu) failed: %s", label, nbytes, strerror(errno));
    if (p == NULL) {
      continue;
    }
    CHECK((uintptr_t)p % page_size == 0, "%s: allocation is not page aligned", label);
    check_vma_policy(p, set, mode, label);

    int* status = malloc(pages * sizeof(int));
    if (status == NULL || residency(p, pages, page_size, status) != 0) {
      CHECK(0, "%s: move_pages query failed: %s", label, strerror(errno));
      free(status);
      numa_allocator_free(ctx, p, nbytes);
      continue;
    }
    /* Without prefault nothing may be populated yet, which shows that the
     * policy was applied before first touch. */
    size_t populated = 0;
    for (size_t i = 0; i < pages; i++) {
      populated += status[i] >= 0;
    }
    CHECK(populated == (prefault ? pages : 0), "%s: %zu of %zu pages populated before the first write (prefault=%d)",
          label, populated, pages, prefault);

    memset(p, 0xA5, pages * page_size);
    size_t bad = 0;
    for (size_t i = 0; i < pages * page_size; i += 61) {
      bad += p[i] != 0xA5;
    }
    CHECK(bad == 0, "%s: %zu bytes did not read back", label, bad);

    CHECK(residency(p, pages, page_size, status) == 0, "%s: move_pages query failed: %s", label, strerror(errno));
    size_t per_node[MAX_NODES] = {0};
    size_t off_mask = 0;
    for (size_t i = 0; i < pages; i++) {
      if (status[i] >= 0 && status[i] < MAX_NODES && in_set(set, status[i])) {
        per_node[status[i]]++;
      } else {
        off_mask++;
      }
    }
    /* MPOL_PREFERRED_MANY may legitimately fall back to other nodes. */
    if (mode != NUMA_ALLOC_PREFERRED_MANY) {
      CHECK(off_mask == 0, "%s: %zu of %zu pages are off the node mask", label, off_mask, pages);
    }
    if (mode == NUMA_ALLOC_INTERLEAVE && set->count > 1 && nbytes == BIG_BYTES) {
      /* Huge pages make the spread coarse, so allow a factor of two. */
      size_t ideal = pages / set->count;
      for (size_t i = 0; i < set->count; i++) {
        size_t got = per_node[set->nodes[i]];
        CHECK(got >= ideal / 2 && got <= ideal * 2, "%s: node %d holds %zu pages, ideal %zu", label,
              set->nodes[i], got, ideal);
      }
    }
    if (nbytes == BIG_BYTES) {
      printf("  %s: %zu pages", label, pages);
      for (size_t i = 0; i < set->count; i++) {
        printf(" node%d=%zu", set->nodes[i], per_node[set->nodes[i]]);
      }
      printf(" off-mask=%zu\n", off_mask);
    }
    free(status);
    numa_allocator_free(ctx, p, nbytes);
  }
}

static void test_rejects_bad_config(const struct node_set* good) {
  char err[256];
  struct numa_alloc_config cfg = {.nodes = good->nodes, .num_nodes = 0};
  CHECK(numa_allocator_create(&cfg, err, sizeof(err)) == NULL && err[0] != '\0', "empty node list accepted");
  CHECK(numa_allocator_create(NULL, err, sizeof(err)) == NULL && err[0] != '\0', "NULL config accepted");

  int negative[] = {-1};
  cfg = (struct numa_alloc_config){.nodes = negative, .num_nodes = 1};
  CHECK(numa_allocator_create(&cfg, err, sizeof(err)) == NULL && err[0] != '\0', "negative node accepted");

  int missing[] = {100000};
  cfg = (struct numa_alloc_config){.nodes = missing, .num_nodes = 1};
  CHECK(numa_allocator_create(&cfg, err, sizeof(err)) == NULL && strstr(err, "not online") != NULL,
        "node 100000 accepted, err: %s", err);

  cfg = (struct numa_alloc_config){.nodes = good->nodes, .num_nodes = good->count, .mode = 99};
  CHECK(numa_allocator_create(&cfg, err, sizeof(err)) == NULL && err[0] != '\0', "unknown mode accepted");

  cfg = (struct numa_alloc_config){.nodes = good->nodes, .num_nodes = good->count, .prefault = 2};
  CHECK(numa_allocator_create(&cfg, err, sizeof(err)) == NULL && err[0] != '\0', "prefault=2 accepted");

  CHECK(numa_allocator_create(&cfg, NULL, 0) == NULL, "prefault=2 accepted without an error buffer");
}

static void test_failed_allocation(void* ctx) {
  errno = 0;
  CHECK(numa_allocator_alloc(ctx, SIZE_MAX - 100) == NULL && errno == ENOMEM, "size overflow not rejected, errno %d", errno);
  errno = 0;
  CHECK(numa_allocator_alloc(ctx, (size_t)1 << 60) == NULL && errno != 0, "impossible mapping not rejected, errno %d", errno);
  errno = 0;
  CHECK(numa_allocator_alloc(NULL, 16) == NULL && errno == EINVAL, "NULL context not rejected");
  numa_allocator_free(ctx, NULL, 16);
  numa_allocator_free(NULL, NULL, 0);
  numa_allocator_destroy(NULL);
}

/* Allocations of one instance stay valid while another instance allocates,
 * frees and is destroyed. */
static void test_independent_instances(void* a, void* b) {
  size_t n = 1 << 20;
  unsigned char* a1 = numa_allocator_alloc(a, n);
  unsigned char* b1 = numa_allocator_alloc(b, n);
  unsigned char* a2 = numa_allocator_alloc(a, n);
  CHECK(a1 && b1 && a2, "allocation failed: %s", strerror(errno));
  if (!(a1 && b1 && a2)) {
    return;
  }
  CHECK(a1 != b1 && a1 != a2 && b1 != a2, "allocations overlap");
  memset(a1, 1, n);
  memset(b1, 2, n);
  memset(a2, 3, n);
  numa_allocator_free(a, a1, n);
  numa_allocator_free(a, a2, n);
  numa_allocator_destroy(a);
  CHECK(b1[0] == 2 && b1[n - 1] == 2, "instance B data changed after instance A was destroyed");
  numa_allocator_free(b, b1, n);
}

int main(int argc, char** argv) {
  struct node_set set_a, set_b;
  if (parse_nodes(argc > 1 ? argv[1] : "0", &set_a) != 0 || parse_nodes(argc > 2 ? argv[2] : "1", &set_b) != 0) {
    fprintf(stderr, "usage: %s [NODES_A [NODES_B]]\n", argv[0]);
    return 2;
  }
  char err[256];

  CHECK(numa_allocator_abi_version() == NUMA_ALLOCATOR_ABI_VERSION, "ABI version %d", numa_allocator_abi_version());

  void* a = create(&set_a, NUMA_ALLOC_BIND, 0, err, sizeof(err));
  if (a == NULL) {
    fprintf(stderr, "FAIL: instance A (nodes %s) is not usable: %s\n", argc > 1 ? argv[1] : "0", err);
    return 1;
  }
  test_rejects_bad_config(&set_a);
  test_failed_allocation(a);

  int mode_b = NUMA_ALLOC_BIND;
  void* b = create(&set_b, mode_b, 1, err, sizeof(err));
  if (b == NULL) {
    printf("SKIP second node set (%s): falling back to instance A's nodes with MPOL_INTERLEAVE\n", err);
    skips++;
    set_b = set_a;
    mode_b = NUMA_ALLOC_INTERLEAVE;
    b = create(&set_b, mode_b, 1, err, sizeof(err));
  }
  CHECK(b != NULL, "instance B: %s", err);

  exercise(a, &set_a, NUMA_ALLOC_BIND, 0, "A bind");
  if (b != NULL) {
    exercise(b, &set_b, mode_b, 1, mode_b == NUMA_ALLOC_BIND ? "B bind prefault" : "B interleave prefault");
  }

  void* c = create(&set_a, NUMA_ALLOC_INTERLEAVE, 0, err, sizeof(err));
  CHECK(c != NULL, "interleave instance: %s", err);
  if (c != NULL) {
    exercise(c, &set_a, NUMA_ALLOC_INTERLEAVE, 0, "A interleave");
    numa_allocator_destroy(c);
  }

  void* d = create(&set_a, NUMA_ALLOC_PREFERRED_MANY, 0, err, sizeof(err));
  if (d == NULL && errno == EINVAL) {
    printf("SKIP MPOL_PREFERRED_MANY: %s\n", err);
    skips++;
  } else {
    CHECK(d != NULL, "preferred-many instance: %s", err);
  }
  if (d != NULL) {
    exercise(d, &set_a, NUMA_ALLOC_PREFERRED_MANY, 0, "A preferred-many");
    numa_allocator_destroy(d);
  }

  if (b != NULL) {
    test_independent_instances(a, b);
    numa_allocator_destroy(b);
  } else {
    numa_allocator_destroy(a);
  }

  printf("%s: %d failure(s), %d skip(s)\n", failures == 0 ? "PASS" : "FAIL", failures, skips);
  return failures == 0 ? 0 : 1;
}
