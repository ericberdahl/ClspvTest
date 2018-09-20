//
// Created by Eric Berdahl on 10/22/17.
//

#include "file_utils.hpp"

#include "util.hpp" // AndroidFopen

#include <cstdio>

namespace file_utils {

    UniqueFILE fopen_unique(const char *filename, const char *mode) {
        return UniqueFILE(AndroidFopen(filename, mode), &std::fclose);
    }

    template<>
    void* get_data_hack(std::string &c)
    {
        return const_cast<char *>(c.data());
    }

}   // namespace file_utils
