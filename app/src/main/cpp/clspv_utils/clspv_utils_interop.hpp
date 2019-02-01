// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_CLSPV_UTILS_INTEROP_HPP
#define CLSPVUTILS_CLSPV_UTILS_INTEROP_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace clspv_utils {

    template <typename T, typename U>
    using map = std::map<T, U>;

    template <typename T>
    using shared_ptr = std::shared_ptr<T>;

    using string = std::string;

    template <typename T>
    using vector = std::vector<T>;

    void fail_runtime_error(const string& what);

    void fail_runtime_error(const char* what);

}

#endif //CLSPVUTILS_CLSPV_UTILS_INTEROP_HPP
