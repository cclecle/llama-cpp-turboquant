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
// cross-process advisory lock
//
// A sidecar "<path>.lock" is used rather than the state file itself so that the lock survives the
// truncate-and-rewrite of the target, and so a reader can hold a shared lock while a save is
// preparing a different file in the same directory.
//

namespace {

class slot_file_lock {
public:
    slot_file_lock(const std::string & path, bool exclusive) {
        const std::string lock_path = path + ".lock";

#if defined(_WIN32)
        handle = CreateFileA(lock_path.c_str(), GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            return;
        }

        OVERLAPPED ov = {};
        const DWORD flags = exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0;
        if (LockFileEx(handle, flags, 0, MAXDWORD, MAXDWORD, &ov)) {
            locked = true;
        }
#else
        fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) {
            return;
        }

        struct flock fl = {};
        fl.l_type   = exclusive ? F_WRLCK : F_RDLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start  = 0;
        fl.l_len    = 0;

        // F_SETLKW blocks until the holder is done. That is fine here and only here: this runs on
        // an I/O worker, never on the inference thread.
        while (::fcntl(fd, F_SETLKW, &fl) == -1) {
            if (errno != EINTR) {
                return;
            }
        }
        locked = true;
#endif
    }

    ~slot_file_lock() {
#if defined(_WIN32)
        if (handle != INVALID_HANDLE_VALUE) {
            if (locked) {
                OVERLAPPED ov = {};
                UnlockFileEx(handle, 0, MAXDWORD, MAXDWORD, &ov);
            }
            CloseHandle(handle);
        }
#else
        if (fd >= 0) {
            if (locked) {
                struct flock fl = {};
                fl.l_type   = F_UNLCK;
                fl.l_whence = SEEK_SET;
                ::fcntl(fd, F_SETLK, &fl);
            }
            ::close(fd);
        }
#endif
    }

    slot_file_lock(const slot_file_lock &)             = delete;
    slot_file_lock & operator=(const slot_file_lock &) = delete;

private:
    // A lock we could not take is not fatal - the sidecar may sit on a filesystem without
    // locking (some network mounts). The caller then proceeds unserialised, which is no worse
    // than the behaviour before locking existed.
    bool locked = false;
#if defined(_WIN32)
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

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

} // namespace

bool slot_file_write(const std::string & path, const std::vector<uint8_t> & bytes, std::string & err) {
    slot_file_lock lock(path, true);

    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        err = "failed to open '" + path + "' for writing: " + std::strerror(errno);
        return false;
    }

    bool ok = true;

    if (!bytes.empty()) {
        ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
        if (!ok) {
            err = "short write to '" + path + "': " + std::strerror(errno);
        }
    }

    // durable before the caller is told the save succeeded
    if (ok && !file_sync(f)) {
        ok  = false;
        err = "failed to flush '" + path + "': " + std::strerror(errno);
    }

    if (std::fclose(f) != 0 && ok) {
        ok  = false;
        err = "failed to close '" + path + "': " + std::strerror(errno);
    }

    if (ok) {
        dir_sync(path);
    }

    return ok;
}

bool slot_file_read(const std::string & path, std::vector<uint8_t> & bytes, std::string & err) {
    slot_file_lock lock(path, false);

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
