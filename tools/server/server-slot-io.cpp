#include "server-slot-io.h"

#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   include <windows.h>
#   include <io.h>
#else
#   include <fcntl.h>
#   include <unistd.h>
#   include <sys/stat.h>
#   include <sys/types.h>
#endif

//
// bounded byte cursors
//

void slot_byte_writer::raw(const void * src, size_t len) {
    if (len == 0) {
        return;
    }
    const uint8_t * p = static_cast<const uint8_t *>(src);
    buf.insert(buf.end(), p, p + len);
}

bool slot_byte_reader::raw(void * dst, size_t len) {
    if (bad || len > n - off) {
        bad = true;
        return false;
    }
    std::memcpy(dst, data + off, len);
    off += len;
    return true;
}

bool slot_byte_reader::seek(size_t pos) {
    if (bad || pos > n) {
        bad = true;
        return false;
    }
    off = pos;
    return true;
}

//
// atomic file replacement
//
// A save writes a temp file in the same directory and renames it over the target. rename is
// atomic, so a reader never observes a partial file and no cross-process lock is needed. It also
// means a crash part-way through a save leaves the previous good state file untouched - only a
// stray ".tmp" is left behind, never a truncated target.
//

namespace {

bool file_sync(FILE * f) {
    if (std::fflush(f) != 0) {
        return false;
    }
#if defined(_WIN32)
    return _commit(_fileno(f)) == 0;
#else
    return ::fsync(::fileno(f)) == 0;
#endif
}

// On POSIX a newly created file is only durable once its directory entry is too.
void dir_sync(const std::string & path) {
#if !defined(_WIN32)
    const size_t slash = path.find_last_of("/\\");
    const std::string dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);

    const int fd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
#else
    (void) path;
#endif
}

// unique per process, so two servers sharing a --slot-save-path cannot clobber each other's temp
std::string temp_path(const std::string & path) {
#if defined(_WIN32)
    const unsigned long pid = GetCurrentProcessId();
#else
    const unsigned long pid = (unsigned long) ::getpid();
#endif
    return path + ".tmp." + std::to_string(pid);
}

bool rename_over(const std::string & from, const std::string & to) {
#if defined(_WIN32)
    // std::rename fails when the target exists, MoveFileEx replaces it
    return MoveFileExA(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return ::rename(from.c_str(), to.c_str()) == 0;
#endif
}

void remove_file(const std::string & path) {
#if defined(_WIN32)
    DeleteFileA(path.c_str());
#else
    ::unlink(path.c_str());
#endif
}

} // namespace

bool slot_file_write(const std::string & path, const std::vector<uint8_t> & bytes, std::string & err) {
    const std::string tmp = temp_path(path);

    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "failed to open '" + tmp + "' for writing: " + std::strerror(errno);
        return false;
    }

    bool ok = true;

    if (!bytes.empty()) {
        ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
        if (!ok) {
            err = "short write to '" + tmp + "': " + std::strerror(errno);
        }
    }

    // the temp must be durable before it is renamed over the target, else a crash just after the
    // rename could leave the target pointing at data that never reached the disk
    if (ok && !file_sync(f)) {
        ok  = false;
        err = "failed to flush '" + tmp + "': " + std::strerror(errno);
    }

    if (std::fclose(f) != 0 && ok) {
        ok  = false;
        err = "failed to close '" + tmp + "': " + std::strerror(errno);
    }

    if (ok && !rename_over(tmp, path)) {
        ok  = false;
        err = "failed to rename '" + tmp + "' onto '" + path + "': " + std::strerror(errno);
    }

    if (!ok) {
        remove_file(tmp);
        return false;
    }

    // the rename itself is a directory change, so the directory entry needs syncing too
    dir_sync(path);

    return true;
}

bool slot_file_read(const std::string & path, std::vector<uint8_t> & bytes, std::string & err) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "failed to open '" + path + "': " + std::strerror(errno);
        return false;
    }

    bool ok = std::fseek(f, 0, SEEK_END) == 0;

    long size = ok ? std::ftell(f) : -1;
    if (size < 0) {
        ok  = false;
        err = "failed to size '" + path + "': " + std::strerror(errno);
    }

    if (ok) {
        ok = std::fseek(f, 0, SEEK_SET) == 0;
    }

    if (ok) {
        bytes.resize((size_t) size);
        if (size > 0 && std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
            ok  = false;
            err = "short read from '" + path + "'";
        }
    }

    std::fclose(f);

    return ok;
}

//
// worker pool
//

struct slot_io_pool::impl {
    std::mutex                        mutex;
    std::condition_variable           cv_job;
    std::deque<std::function<void()>> jobs;
    std::vector<std::thread>          threads;
    size_t                            max_pending = 0;
    bool                              running     = true;

    void loop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_job.wait(lock, [&] { return !running || !jobs.empty(); });
                if (!running && jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }

            // a throwing job must not take the whole pool down with it
            try {
                job();
            } catch (...) {
            }
        }
    }
};

slot_io_pool::slot_io_pool(int n_threads, size_t max_pending) : pimpl(new impl()) {
    pimpl->max_pending = max_pending;
    for (int i = 0; i < n_threads; ++i) {
        pimpl->threads.emplace_back([this] { pimpl->loop(); });
    }
}

slot_io_pool::~slot_io_pool() {
    shutdown();
}

bool slot_io_pool::submit(std::function<void()> job) {
    {
        std::unique_lock<std::mutex> lock(pimpl->mutex);
        if (!pimpl->running || pimpl->jobs.size() >= pimpl->max_pending) {
            return false;
        }
        pimpl->jobs.push_back(std::move(job));
    }
    pimpl->cv_job.notify_one();
    return true;
}

void slot_io_pool::shutdown() {
    {
        std::unique_lock<std::mutex> lock(pimpl->mutex);
        if (!pimpl->running) {
            return;
        }
        pimpl->running = false;
    }
    pimpl->cv_job.notify_all();

    for (auto & t : pimpl->threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    pimpl->threads.clear();
}
