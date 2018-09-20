//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVTEST_FILE_UTILS_HPP
#define CLSPVTEST_FILE_UTILS_HPP

#include "file_utils.hpp"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>

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

    template<typename Container>
    void read_file_contents(const std::string &filename, Container &fileContents) {
        const std::size_t wordSize = sizeof(typename Container::value_type);

        UniqueFILE pFile = fopen_unique(filename.c_str(), "rb");
        if (!pFile) {
            throw std::runtime_error("can't open file: " + filename);
        }

        std::fseek(pFile.get(), 0, SEEK_END);

        const auto num_bytes = std::ftell(pFile.get());
        if (0 != (num_bytes % wordSize)) {
            throw std::runtime_error(
                    "file size of " + filename + " inappropriate for requested type");
        }

        const auto num_words = (num_bytes + wordSize - 1) / wordSize;
        fileContents.resize(num_words);
        assert(num_bytes == (fileContents.size() * wordSize));

        std::fseek(pFile.get(), 0, SEEK_SET);
        std::fread(get_data_hack(fileContents), 1, num_bytes, pFile.get());
    }

}   // namespace file_utils

#endif // CLSPVTEST_FILE_UTILS_HPP