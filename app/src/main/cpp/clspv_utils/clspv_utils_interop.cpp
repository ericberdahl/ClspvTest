//
// Created by Eric Berdahl on 9/20/18.
//

#include "clspv_utils_interop.hpp"

#include <stdexcept>

namespace clspv_utils {

    void fail_runtime_error(const char* what)
    {
        throw std::runtime_error(what);
    }

    void fail_runtime_error(const string& what)
    {
        throw std::runtime_error(what);
    }

}
