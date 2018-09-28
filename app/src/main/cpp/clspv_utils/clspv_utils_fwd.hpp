//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_CLSPV_UTILS_FWD_HPP
#define CLSPVUTILS_CLSPV_UTILS_FWD_HPP

namespace clspv_utils {

    // execution types
    class device;
    class invocation;
    class kernel;
    class module;

    struct execution_time_t;
    struct kernel_req_t;
    struct invocation_req_t;

    // interface types
    struct arg_spec_t;
    struct kernel_spec_t;
    struct module_spec_t;
    struct sampler_spec_t;

}

#endif //CLSPVUTILS_CLSPV_UTILS_FWD_HPP
