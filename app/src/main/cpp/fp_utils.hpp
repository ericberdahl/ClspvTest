//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_FP_UTILS_HPP
#define CLSPVTEST_FP_UTILS_HPP

#include <cstdlib>
#include <limits>
#include <type_traits>

namespace fp_utils {
    template<class T>
    typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp) {
        // the machine epsilon has to be scaled to the magnitude of the values used
        // and multiplied by the desired precision in ULPs (units in the last place)
        return std::abs(x - y) < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
               // unless the result is subnormal
               || std::abs(x - y) < std::numeric_limits<T>::min();
    }
}

#endif //CLSPVTEST_FP_UTILS_HPP
