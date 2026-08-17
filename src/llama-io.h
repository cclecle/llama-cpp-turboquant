#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

struct ggml_tensor;

// Thrown by the memory implementations when a state cannot be restored, carrying a
// machine-readable reason. The llama_memory_i::state_read signature stays unchanged, so every
// implementation (unified KV, iSWA, recurrent, hybrid) can report accurately without plumbing
// an out-parameter through the interface.
class llama_state_seq_error : public std::runtime_error {
public:
    llama_state_seq_error(llama_state_seq_status status, const std::string & msg) :
        std::runtime_error(msg), status(status) {}

    llama_state_seq_status status;
};

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
