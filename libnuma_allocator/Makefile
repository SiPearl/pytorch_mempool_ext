CC ?= cc
CFLAGS ?= -O2 -g
# Kept apart from CFLAGS so that `make CFLAGS=...` cannot drop -fPIC.
ALL_CFLAGS := -std=c11 -Wall -Wextra -Werror -fPIC -fvisibility=hidden $(CFLAGS)
BUILD ?= build

# Node lists for `make test`, for example TEST_NODES="0-3 4-7" on Rhea.
TEST_NODES ?= 0 1

have_mempolicy := $(shell $(CC) -E -include linux/mempolicy.h -x c /dev/null >/dev/null 2>&1 && echo yes)
ifneq ($(have_mempolicy),yes)
$(error <linux/mempolicy.h> not found; the Linux UAPI headers are required)
endif

LIB := $(BUILD)/libnuma_allocator.so
TEST := $(BUILD)/test_numa_allocator

all: $(LIB) $(TEST)

$(BUILD):
	mkdir -p $@

# -Bsymbolic binds calls inside the library to its own definitions, whatever
# else is loaded in the process.
$(LIB): numa_allocator.c numa_allocator.h | $(BUILD)
	$(CC) $(ALL_CFLAGS) -shared -Wl,-Bsymbolic -Wl,-soname,libnuma_allocator.so -o $@ numa_allocator.c

$(TEST): test_numa_allocator.c numa_allocator.h $(LIB)
	$(CC) $(ALL_CFLAGS) -o $@ test_numa_allocator.c -L$(BUILD) -lnuma_allocator -Wl,-rpath,'$$ORIGIN'

test: $(TEST)
	$(TEST) $(TEST_NODES)

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
