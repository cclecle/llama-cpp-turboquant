#pragma once

// Off-thread slot state I/O. [TAG_SLOT_IO_ASYNC]
//
// Saving or restoring a slot moves 100-250 MB against --slot-save-path. Doing that inline in the
// task loop freezes every other slot for the whole transfer, so the work is split in two:
//
//   - the device <-> host copy of the KV bytes stays on the inference thread, because llama.cpp
//     backends cannot be driven from two threads at once. It is a few ms.
//   - the file read/write, the fsync and the cross-process lock run on a worker thread here.
//     That is the part measured in seconds.
//
// Workers therefore never touch llama_context: they only ever see the finished byte image.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

//
// bounded byte cursors
//
// The slot file trailer used to be written with bare fwrite/fread whose return values were mostly
// unchecked. These are drop-in replacements that cannot run off the end of the buffer: a reader
// that hits the end latches ok=false and every later read is a no-op, so a truncated file fails
// the final ok() check instead of parsing garbage.
//

struct slot_byte_writer {
    std::vector<uint8_t> buf;

    void raw(const void * src, size_t len);

    template <typename T> void pod(const T & v) {
        static_assert(std::is_trivially_copyable<T>::value, "pod() needs a trivially copyable type");
        raw(&v, sizeof(v));
    }

    size_t size() const { return buf.size(); }
};

struct slot_byte_reader {
    slot_byte_reader(const uint8_t * data, size_t size) : data(data), n(size) {}

    const uint8_t * data;
    size_t          n;
    size_t          off = 0;
    bool            bad = false;

    bool raw(void * dst, size_t len);

    template <typename T> bool pod(T & v) {
        static_assert(std::is_trivially_copyable<T>::value, "pod() needs a trivially copyable type");
        return raw(&v, sizeof(v));
    }

    // absolute seek; out-of-range latches the failure like a short read does
    bool seek(size_t pos);

    bool   ok()        const { return !bad; }
    size_t remaining() const { return bad ? 0 : n - off; }
};

//
// file access
//
// Both calls take an advisory cross-process lock on "<path>.lock" for their duration - exclusive
// for a write, shared for a read - because several server processes can share one
// --slot-save-path and nothing else coordinates them. They block while waiting, which is safe
// only because they run on a worker thread. Never call them from the inference thread.
//

// Write the whole image, fsync it, then release the lock. Returns false and fills err on failure.
bool slot_file_write(const std::string & path, const std::vector<uint8_t> & bytes, std::string & err);

// Read the whole file. Returns false and fills err on failure.
bool slot_file_read(const std::string & path, std::vector<uint8_t> & bytes, std::string & err);

//
// worker pool
//
// Jobs run on a small pool and report back by posting to the server queues, which are
// mutex-guarded and therefore safe to touch from here.
//

class slot_io_pool {
public:
    // n_threads workers; in-flight jobs are capped so a burst of concurrent promotions cannot
    // balloon host memory with several hundred MB of state each
    explicit slot_io_pool(int n_threads = 2, size_t max_pending = 8);
    ~slot_io_pool();

    slot_io_pool(const slot_io_pool &)             = delete;
    slot_io_pool & operator=(const slot_io_pool &) = delete;

    // returns false if the pool is shutting down or already has max_pending jobs queued
    bool submit(std::function<void()> job);

    void shutdown();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
