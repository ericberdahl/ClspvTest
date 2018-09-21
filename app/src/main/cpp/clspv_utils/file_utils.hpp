//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVTEST_FILE_UTILS_HPP
#define CLSPVTEST_FILE_UTILS_HPP

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <streambuf>
#include <vector>

namespace file_utils {

    typedef std::unique_ptr<std::FILE, decltype(&std::fclose)> UniqueFILE;

    UniqueFILE fopen_unique(const char *filename, const char *mode);

    //
    // get_data_hack works around a deficiency in std::string, prior to C++17, in which
    // std::string::data() only returns a const char*.
    //
    template<typename Container>
    void *get_data_hack(Container &c) { return c.data(); }

    template<>
    void *get_data_hack(std::string &c);

    class FILE_buffer : public std::streambuf
    {
    public:
                        FILE_buffer();

        explicit        FILE_buffer(FILE* fptr, std::size_t buff_sz = 256, std::size_t put_back = 8);

                        FILE_buffer(const FILE_buffer &) = delete;

                        FILE_buffer(FILE_buffer &&);

        FILE_buffer&    operator= (const FILE_buffer &) = delete;

        FILE_buffer&    operator= (FILE_buffer &&);

        void            swap(FILE_buffer& other);

    private:
        virtual int_type underflow();

    protected:
        virtual pos_type seekoff( off_type off, std::ios_base::seekdir dir,
                                  std::ios_base::openmode which = std::ios_base::in | std::ios_base::out );

    private:
        FILE*               mFile;
        std::size_t         mPutBack;
        std::vector<char>   mBuffer;
    };

    class AndroidAssetStream : public std::istream {
    public:
                            AndroidAssetStream();

        explicit            AndroidAssetStream(const char* filename);

                            AndroidAssetStream(const AndroidAssetStream &) = delete;

                            AndroidAssetStream(AndroidAssetStream &&);

        AndroidAssetStream& operator= (const AndroidAssetStream &) = delete;

        AndroidAssetStream& operator= (AndroidAssetStream &&);

        bool                is_open() const;

        void                open(const char* filename);

        void                close();

        void                swap(AndroidAssetStream& other);

    private:
        UniqueFILE  mAssetFile;
        FILE_buffer mStreamBuf;
    };

    template<typename Container>
    void read_file_contents(const std::string &filename, Container &fileContents) {
        const std::size_t wordSize = sizeof(typename Container::value_type);

        AndroidAssetStream in(filename.c_str());
        if (!in.good())
        {
            throw std::runtime_error("can't open file: " + filename);
        }

        in.seekg(0, std::ios_base::end);

        const auto num_bytes = in.tellg();
        if (0 != (num_bytes % wordSize)) {
            throw std::runtime_error(
                    "file size of " + filename + " inappropriate for requested type");
        }

        const auto num_words = (num_bytes + std::streamoff(wordSize - 1)) / wordSize;
        fileContents.resize(num_words);
        assert(num_bytes == (fileContents.size() * wordSize));

        in.seekg(0, std::ios_base::beg);
        in.read(static_cast<char*>(get_data_hack(fileContents)), num_bytes);
    }

}   // namespace file_utils

#endif // CLSPVTEST_FILE_UTILS_HPP