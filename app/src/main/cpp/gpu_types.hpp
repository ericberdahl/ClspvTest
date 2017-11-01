//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_GPU_TYPES_HPP
#define CLSPVTEST_GPU_TYPES_HPP

#include "half.hpp"

#include <utility>

namespace gpu_types {
    template<typename T>
    struct alignas(2 * sizeof(T)) vec2 {
        vec2(T a, T b) : x(a), y(b) {}

        vec2(T a) : vec2(a, T(0)) {}

        vec2() : vec2(T(0), T(0)) {}

        vec2(const vec2<T> &other) : x(other.x), y(other.y) {}

        vec2(vec2<T> &&other) : vec2() {
            swap(*this, other);
        }

        vec2<T> &operator=(vec2<T> other) {
            swap(*this, other);
            return *this;
        }

        T x;
        T y;
    };

    template<typename T>
    void swap(vec2<T> &first, vec2<T> &second) {
        using std::swap;

        swap(first.x, second.x);
        swap(first.y, second.y);
    }

    template<typename T>
    bool operator==(const vec2<T> &l, const vec2<T> &r) {
        return (l.x == r.x) && (l.y == r.y);
    }

    template<typename T>
    struct alignas(4 * sizeof(T)) vec4 {
        vec4(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}

        vec4(T a, T b, T c) : vec4(a, b, c, T(0)) {}

        vec4(T a, T b) : vec4(a, b, T(0), T(0)) {}

        vec4(T a) : vec4(a, T(0), T(0), T(0)) {}

        vec4() : vec4(T(0), T(0), T(0), T(0)) {}

        vec4(const vec4<T> &other) : x(other.x), y(other.y), z(other.z), w(other.w) {}

        vec4(vec4<T> &&other) : vec4() {
            swap(*this, other);
        }

        vec4<T> &operator=(vec4<T> other) {
            swap(*this, other);
            return *this;
        }

        T x;
        T y;
        T z;
        T w;
    };

    template<typename T>
    void swap(vec4<T> &first, vec4<T> &second) {
        using std::swap;

        swap(first.x, second.x);
        swap(first.y, second.y);
        swap(first.z, second.z);
        swap(first.w, second.w);
    }

    template<typename T>
    bool operator==(const vec4<T> &l, const vec4<T> &r) {
        return (l.x == r.x) && (l.y == r.y) && (l.z == r.z) && (l.w == r.w);
    }

    static_assert(sizeof(float) == 4, "bad size for float");

    typedef vec2<float> float2;
    static_assert(sizeof(float2) == 8, "bad size for float2");

    template<>
    bool operator==(const float2 &l, const float2 &r);

    typedef vec4<float> float4;
    static_assert(sizeof(float4) == 16, "bad size for float4");

    template<>
    bool operator==(const float4 &l, const float4 &r);

    typedef half_float::half half;
    static_assert(sizeof(half) == 2, "bad size for half");

    typedef vec2<half> half2;
    static_assert(sizeof(half2) == 4, "bad size for half2");

    typedef vec4<half> half4;
    static_assert(sizeof(half4) == 8, "bad size for half4");

    typedef unsigned short ushort;
    static_assert(sizeof(ushort) == 2, "bad size for ushort");

    typedef vec2<ushort> ushort2;
    static_assert(sizeof(ushort2) == 4, "bad size for ushort2");

    typedef vec4<ushort> ushort4;
    static_assert(sizeof(ushort4) == 8, "bad size for ushort4");

    typedef unsigned char uchar;
    static_assert(sizeof(uchar) == 1, "bad size for uchar");

    typedef vec2<uchar> uchar2;
    static_assert(sizeof(uchar2) == 2, "bad size for uchar2");

    typedef vec4<uchar> uchar4;
    static_assert(sizeof(uchar4) == 4, "bad size for uchar4");
}

template<>
struct std::is_floating_point<gpu_types::half> : std::true_type {
};
static_assert(std::is_floating_point<gpu_types::half>::value, "half should be floating point");

#endif //CLSPVTEST_GPU_TYPES_HPP
