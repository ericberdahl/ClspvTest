//
// Created by Eric Berdahl on 10/22/17.
//

#ifndef CLSPVUTILS_ARG_SPEC_HPP
#define CLSPVUTILS_ARG_SPEC_HPP

#include "clspv_utils_fwd.hpp"

namespace clspv_utils {

    struct arg_spec_t {
        enum kind_t {
            kind_unknown,
            kind_pod,
            kind_pod_ubo,
            kind_buffer,
            kind_ro_image,
            kind_wo_image,
            kind_sampler,
            kind_local
        };

        kind_t  kind            = kind_unknown;
        int     ordinal         = -1;
        int     descriptor_set  = -1;
        int     binding         = -1;
        int     offset          = -1;
        int     spec_constant   = -1;
    };
}

#endif //CLSPVUTILS_ARG_SPEC_HPP
