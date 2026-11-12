#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/EmptyTensor.h>
#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/CPUAllocatorOverride.h>

#include <numa_allocator.h>

#include <dlfcn.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

struct Counters {
  std::atomic<int64_t> create_calls{0};
  std::atomic<int64_t> destroy_calls{0};
  // Successful numa_allocator_alloc calls. Zero-byte allocations never reach
  // the library, so they count in live_allocations only.
  std::atomic<int64_t> alloc_calls{0};
  std::atomic<int64_t> free_calls{0};
  std::atomic<int64_t> live_allocations{0};
  std::atomic<int64_t> live_bytes{0};
};

struct SymbolNames {
  std::string abi_version;
  std::string create;
  std::string alloc;
  std::string free;
  std::string destroy;
};

struct DlCloser {
  void operator()(void* handle) const {
    dlclose(handle);
  }
};

class State;

// Never freed. A storage keeps a raw pointer to its allocator, even after
// untyped_storage().resize_(0) has dropped the DataPtr that kept the state
// alive, so the object must outlive the state. allocate() raises once the
// state is gone.
class NumaAllocator final : public at::Allocator {
 public:
  explicit NumaAllocator(std::weak_ptr<State> state) : state_(std::move(state)) {}

  at::DataPtr allocate(size_t nbytes) override;

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    default_copy_data(dest, src, count);
  }

 private:
  const std::weak_ptr<State> state_;
};

// Owns the dlopen handle and the C context. The Python handle, every live
// allocation and every active override scope hold a strong reference, so
// numa_allocator_destroy and dlclose run only when all of them are gone.
class State {
 public:
  static std::shared_ptr<State> create(
      const std::string& lib_path,
      const SymbolNames& names,
      const std::vector<int>& nodes,
      int mode,
      bool prefault) {
    auto state = std::make_shared<State>(lib_path, names, nodes, mode, prefault);
    state->allocator_ = new NumaAllocator(state);
    return state;
  }

  // Use create(): the allocator needs a weak reference to the finished object.
  State(
      const std::string& lib_path,
      const SymbolNames& names,
      const std::vector<int>& nodes,
      int mode,
      bool prefault)
      : counters_(std::make_shared<Counters>()) {
    lib_.reset(dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL));
    if (!lib_) {
      throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    }
    auto abi_version = symbol<int (*)()>(names.abi_version);
    auto create = symbol<void* (*)(const numa_alloc_config*, char*, size_t)>(names.create);
    alloc_ = symbol<void* (*)(void*, size_t)>(names.alloc);
    free_ = symbol<void (*)(void*, void*, size_t)>(names.free);
    destroy_ = symbol<void (*)(void*)>(names.destroy);

    int version = abi_version();
    if (version != NUMA_ALLOCATOR_ABI_VERSION) {
      throw std::runtime_error(
          lib_path + " has ABI version " + std::to_string(version) + ", expected " +
          std::to_string(NUMA_ALLOCATOR_ABI_VERSION));
    }
    numa_alloc_config cfg{nodes.data(), nodes.size(), mode, prefault ? 1 : 0};
    char err[256] = "";
    ctx_ = create(&cfg, err, sizeof(err));
    if (ctx_ == nullptr) {
      throw std::runtime_error(names.create + " failed: " + err);
    }
    counters_->create_calls++;
  }

  ~State() {
    destroy_(ctx_);
    counters_->destroy_calls++;
  }

  State(const State&) = delete;
  State& operator=(const State&) = delete;

  at::Allocator* allocator() {
    return allocator_;
  }

  const std::shared_ptr<Counters>& counters() const {
    return counters_;
  }

  void* acquire(size_t nbytes) {
    void* ptr = nullptr;
    if (nbytes != 0) {
      ptr = alloc_(ctx_, nbytes);
      int alloc_errno = errno;
      TORCH_CHECK(
          ptr != nullptr,
          "numa_allocator: failed to allocate ", nbytes, " bytes: ", std::strerror(alloc_errno));
      counters_->alloc_calls++;
    }
    counters_->live_allocations++;
    counters_->live_bytes += static_cast<int64_t>(nbytes);
    return ptr;
  }

  void release(void* ptr, size_t nbytes) {
    if (ptr != nullptr) {
      free_(ctx_, ptr, nbytes);
      counters_->free_calls++;
    }
    counters_->live_allocations--;
    counters_->live_bytes -= static_cast<int64_t>(nbytes);
  }

 private:
  template <typename Fn>
  Fn symbol(const std::string& name) {
    void* address = dlsym(lib_.get(), name.c_str());
    if (address == nullptr) {
      throw std::runtime_error("symbol not found in the allocator library: " + name);
    }
    return reinterpret_cast<Fn>(address);
  }

  std::shared_ptr<Counters> counters_;
  std::unique_ptr<void, DlCloser> lib_;
  void* (*alloc_)(void*, size_t) = nullptr;
  void (*free_)(void*, void*, size_t) = nullptr;
  void (*destroy_)(void*) = nullptr;
  void* ctx_ = nullptr;
  NumaAllocator* allocator_ = nullptr;
};

// DataPtr context. Zero-byte allocations get one too: their storage keeps a raw
// pointer to the allocator and may be resized later.
struct Block {
  std::shared_ptr<State> state;
  void* ptr;
  size_t nbytes;
};

void delete_block(void* ctx) {
  std::unique_ptr<Block> block(static_cast<Block*>(ctx));
  block->state->release(block->ptr, block->nbytes);
}

at::DataPtr NumaAllocator::allocate(size_t nbytes) {
  auto state = state_.lock();
  TORCH_CHECK(
      state != nullptr,
      "numa_allocator: cannot allocate ", nbytes, " bytes: the allocator was destroyed after its NumaAllocator ",
      "object, its allocations and its use_allocator scopes were all gone");
  auto block = std::make_unique<Block>(Block{std::move(state), nullptr, nbytes});
  block->ptr = block->state->acquire(nbytes);
  void* ptr = block->ptr;
  return {ptr, block.release(), &delete_block, at::Device(at::kCPU)};
}

// One `with use_allocator(...)` block. The core guard restores the previous
// override of whichever thread destroys it, so it may only be destroyed on the
// entering thread while it is the innermost scope.
class OverrideScope {
 public:
  explicit OverrideScope(std::shared_ptr<State> state)
      : thread_(std::this_thread::get_id()), active_(std::make_unique<Active>(std::move(state))) {}

  ~OverrideScope() {
    // Restoring here would corrupt the override of some thread, so the guard is
    // leaked. The state is leaked with it, so that the override left behind
    // keeps allocating.
    if (active_ && !is_innermost()) {
      (void)active_.release();
    }
  }

  void exit() {
    if (!active_) {
      throw std::runtime_error("use_allocator scope was already exited");
    }
    if (!is_innermost()) {
      throw std::runtime_error(
          "use_allocator scopes must be exited in reverse order, on the thread that entered them");
    }
    active_.reset();
  }

 private:
  struct Active {
    explicit Active(std::shared_ptr<State> s)
        : state(std::move(s)), guard(state ? state->allocator() : nullptr) {}
    std::shared_ptr<State> state;
    c10::impl::CPUAllocatorOverrideGuard guard;
  };

  bool is_innermost() const {
    at::Allocator* mine = active_->state ? active_->state->allocator() : nullptr;
    return std::this_thread::get_id() == thread_ && c10::impl::getCPUAllocatorOverride() == mine;
  }

  std::thread::id thread_;
  std::unique_ptr<Active> active_;
};

at::Tensor empty(const std::shared_ptr<State>& state, const std::vector<int64_t>& size, at::ScalarType dtype) {
  c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return at::detail::empty_generic(size, state->allocator(), cpu_ks, dtype, std::nullopt);
}

bool owns(const std::shared_ptr<State>& state, const at::Tensor& tensor) {
  return tensor.defined() && tensor.has_storage() && tensor.storage().allocator() == state->allocator();
}

// Test-only: resize_ of a tensor without storage is one of the two places where
// the core reads the override.
at::Tensor storageless_tensor(at::ScalarType dtype) {
  return at::detail::make_tensor<at::TensorImpl>(
      c10::DispatchKeySet(c10::DispatchKey::CPU), c10::scalarTypeToTypeMeta(dtype), at::Device(at::kCPU));
}

} // namespace

PYBIND11_MODULE(_C, m) {
  m.attr("BIND") = static_cast<int>(NUMA_ALLOC_BIND);
  m.attr("INTERLEAVE") = static_cast<int>(NUMA_ALLOC_INTERLEAVE);
  m.attr("PREFERRED_MANY") = static_cast<int>(NUMA_ALLOC_PREFERRED_MANY);

  py::class_<Counters, std::shared_ptr<Counters>>(m, "Counters")
      .def_property_readonly("create_calls", [](const Counters& c) { return c.create_calls.load(); })
      .def_property_readonly("destroy_calls", [](const Counters& c) { return c.destroy_calls.load(); })
      .def_property_readonly("alloc_calls", [](const Counters& c) { return c.alloc_calls.load(); })
      .def_property_readonly("free_calls", [](const Counters& c) { return c.free_calls.load(); })
      .def_property_readonly("live_allocations", [](const Counters& c) { return c.live_allocations.load(); })
      .def_property_readonly("live_bytes", [](const Counters& c) { return c.live_bytes.load(); });

  py::class_<State, std::shared_ptr<State>>(m, "NumaAllocatorImpl")
      .def(py::init([](const std::string& lib_path,
                       const std::vector<int>& nodes,
                       int mode,
                       bool prefault,
                       const std::string& abi_version_fn_name,
                       const std::string& create_fn_name,
                       const std::string& alloc_fn_name,
                       const std::string& free_fn_name,
                       const std::string& destroy_fn_name) {
        SymbolNames names{abi_version_fn_name, create_fn_name, alloc_fn_name, free_fn_name, destroy_fn_name};
        return State::create(lib_path, names, nodes, mode, prefault);
      }))
      .def_property_readonly("counters", &State::counters)
      .def("empty", torch::wrap_pybind_function_no_gil(&empty))
      .def("owns", torch::wrap_pybind_function(&owns));

  py::class_<OverrideScope>(m, "OverrideScope")
      .def(py::init<std::shared_ptr<State>>())
      .def("exit", &OverrideScope::exit);

  m.def(
      "_storageless_tensor",
      torch::wrap_pybind_function(&storageless_tensor),
      "Test-only. CPU tensor of the given dtype that has no storage yet. Not part of the API.");
}
