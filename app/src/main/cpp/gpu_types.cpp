//
// Created by Eric Berdahl on 10/31/17.
//

#include "gpu_types.hpp"

#include "fp_utils.hpp"

namespace gpu_types {

    template<>
    bool operator==(const float2 &l, const float2 &r) {
        const int ulp = 2;
        return fp_utils::almost_equal(l.x, r.x, ulp)
               && fp_utils::almost_equal(l.y, r.y, ulp);
    }

    template<>
    bool operator==(const float4 &l, const float4 &r) {
        const int ulp = 2;
        return fp_utils::almost_equal(l.x, r.x, ulp)
               && fp_utils::almost_equal(l.y, r.y, ulp)
               && fp_utils::almost_equal(l.z, r.z, ulp)
               && fp_utils::almost_equal(l.w, r.w, ulp);
    }

}