//
// Created by Eric Berdahl on 10/22/17.
//

#include "file_utils.hpp"

#include "util.hpp" // AndroidFopen

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>

namespace file_utils {

    UniqueFILE fopen_unique(const char *filename, const char *mode) {
        return UniqueFILE(AndroidFopen(filename, mode), &std::fclose);
    }

    template<>
    void* get_data_hack(std::string &c)
    {
        return const_cast<char *>(c.data());
    }

    FILE_buffer::FILE_buffer()
            : std::streambuf(),
              mFile(nullptr),
              mPutBack(1),
              mBuffer(2)
    {
    }

    FILE_buffer::FILE_buffer(FILE *fptr, std::size_t buff_sz, std::size_t put_back)
            : FILE_buffer()
    {
        mFile = fptr;
        mPutBack = std::max(put_back, std::size_t(1));
        mBuffer.resize(std::max(buff_sz, mPutBack) + mPutBack);

        char *end = mBuffer.data() + mBuffer.size();
        setg(end, end, end);
    }

    FILE_buffer::FILE_buffer(FILE_buffer&& other)
            : FILE_buffer()
    {
        swap(other);
    }

    FILE_buffer& FILE_buffer::operator=(FILE_buffer&& other)
    {
        swap(other);
        return *this;
    }

    void FILE_buffer::swap(FILE_buffer& other)
    {
        using std::swap;

        std::streambuf::swap(other);

        swap(mFile, other.mFile);
        swap(mPutBack, other.mPutBack);
        swap(mBuffer, other.mBuffer);
    }

    FILE_buffer::int_type FILE_buffer::underflow()
    {
        if (gptr() < egptr()) // buffer not exhausted
            return traits_type::to_int_type(*gptr());

        char *base = mBuffer.data();
        char *start = base;

        if (eback() == base) // true when this isn't the first fill
        {
            // Make arrangements for putback characters
            std::memmove(base, egptr() - mPutBack, mPutBack);
            start += mPutBack;
        }

        // start is now the start of the buffer, proper.
        // Read from mFile in to the provided buffer
        std::size_t n = std::fread(start, 1, mBuffer.size() - (start - base), mFile);
        if (n == 0)
            return traits_type::eof();

        // Set buffer pointers
        setg(base, start, start + n);

        return traits_type::to_int_type(*gptr());
    }

    std::streambuf::pos_type FILE_buffer::seekoff(off_type                  off,
                                                  std::ios_base::seekdir    dir,
                                                  std::ios_base::openmode   which)
    {
        if (0 != (which & std::ios_base::out))
            return pos_type(-1);

        int origin;
        switch (dir) {
            case std::ios_base::beg:
                origin = SEEK_SET;
                break;

            case std::ios_base::cur:
                origin = SEEK_CUR;
                break;

            case std::ios_base::end:
                origin = SEEK_END;
                break;

            default:
                return pos_type(-1);
        }

        std::fseek(mFile, off, origin);
        return pos_type(std::ftell(mFile));
    }

    AndroidAssetStream::AndroidAssetStream()
            : std::istream(&mStreamBuf),
              mAssetFile(nullptr, &fclose),
              mStreamBuf(mAssetFile.get())
    {
    }

    AndroidAssetStream::AndroidAssetStream(const char* filename)
            : AndroidAssetStream()
    {
        open(filename);
    }

    AndroidAssetStream::AndroidAssetStream(const std::string& filename)
            : AndroidAssetStream()
    {
        open(filename.c_str());
    }

    AndroidAssetStream::AndroidAssetStream(AndroidAssetStream && other)
            : AndroidAssetStream()
    {
        swap(other);
    }

    AndroidAssetStream&
    AndroidAssetStream::operator=(AndroidAssetStream && other)
    {
        swap(other);
        return *this;
    }

    bool
    AndroidAssetStream::is_open() const
    {
        return (bool) mAssetFile;
    }

    void
    AndroidAssetStream::open(const char* filename)
    {
        mAssetFile = fopen_unique(filename, "r");
        mStreamBuf = FILE_buffer(mAssetFile.get(), 1024);

        if (is_open())
        {
            clear();
        }
        else
        {
            setstate(failbit);
        }
    }

    void
    AndroidAssetStream::close()
    {
        mAssetFile.reset();
    }

    void
    AndroidAssetStream::swap(AndroidAssetStream& other)
    {
        using std::swap;

        std::istream::swap(other);
        mAssetFile.swap(other.mAssetFile);
        mStreamBuf.swap(other.mStreamBuf);
    }

}   // namespace file_utils
