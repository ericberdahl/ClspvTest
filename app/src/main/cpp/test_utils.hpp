//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_TEST_UTILS_HPP
#define CLSPVTEST_TEST_UTILS_HPP

#include "clspv_utils.hpp"
#include "fp_utils.hpp"
#include "gpu_types.hpp"
#include "pixels.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace test_utils {

    namespace details {
        template<typename ExpectedPixelType, typename ObservedPixelType>
        struct pixel_promotion {
            static constexpr const int expected_vec_size = pixels::traits<ExpectedPixelType>::num_components;
            static constexpr const int observed_vec_size = pixels::traits<ObservedPixelType>::num_components;
            static constexpr const int vec_size = (expected_vec_size > observed_vec_size
                                                   ? observed_vec_size : expected_vec_size);

            typedef typename pixels::traits<ExpectedPixelType>::component_t expected_comp_type;
            typedef typename pixels::traits<ObservedPixelType>::component_t observed_comp_type;

            static constexpr const bool expected_is_smaller =
                    sizeof(expected_comp_type) < sizeof(observed_comp_type);
            typedef typename std::conditional<expected_is_smaller, expected_comp_type, observed_comp_type>::type smaller_comp_type;
            typedef typename std::conditional<!expected_is_smaller, expected_comp_type, observed_comp_type>::type larger_comp_type;

            static constexpr const bool smaller_is_floating = std::is_floating_point<smaller_comp_type>::value;
            typedef typename std::conditional<smaller_is_floating, smaller_comp_type, larger_comp_type>::type comp_type;

            typedef typename pixels::vector<smaller_comp_type, vec_size>::type promotion_type;
        };

        template<typename T>
        struct pixel_comparator {
        };

        template<>
        struct pixel_comparator<std::int32_t> {
            static bool is_equal(std::int32_t l, std::int32_t r) {
                return l == r;
            }
        };

        template<>
        struct pixel_comparator<float> {
            static bool is_equal(float l, float r) {
                const int ulp = 2;
                return fp_utils::almost_equal(l, r, ulp);
            }
        };

        template<>
        struct pixel_comparator<gpu_types::half> {
            static bool is_equal(gpu_types::half l, gpu_types::half r) {
                const int ulp = 2;
                return fp_utils::almost_equal(l, r, ulp);
            }
        };

        template<>
        struct pixel_comparator<gpu_types::uchar> {
            static bool is_equal(gpu_types::uchar l, gpu_types::uchar r) {
                // because rounding modes in Vulkan are undefined, unknowable, and unsettable, we
                // need to tolerate off-by-one differences in integral components
                return std::abs(static_cast<int>(l) - static_cast<int>(r)) <= 1;
            }
        };

        template<typename T>
        struct pixel_comparator<gpu_types::vec2<T> > {
            static bool is_equal(const gpu_types::vec2<T> &l, const gpu_types::vec2<T> &r) {
                return pixel_comparator<T>::is_equal(l.x, r.x)
                       && pixel_comparator<T>::is_equal(l.y, r.y);
            }
        };

        template<typename T>
        struct pixel_comparator<gpu_types::vec4<T> > {
            static bool is_equal(const gpu_types::vec4<T> &l, const gpu_types::vec4<T> &r) {
                return pixel_comparator<T>::is_equal(l.x, r.x)
                       && pixel_comparator<T>::is_equal(l.y, r.y)
                       && pixel_comparator<T>::is_equal(l.z, r.z)
                       && pixel_comparator<T>::is_equal(l.w, r.w);
            }
        };
    }

    struct InvocationResult {
        std::string                     mVariation;
        bool                            mSkipped    = false;
        unsigned int                    mNumCorrect = 0;
        unsigned int                    mNumErrors  = 0;
        std::vector<std::string>        mMessages;
        clspv_utils::execution_time_t   mExecutionTime;
    };

    typedef std::vector<InvocationResult> InvocationResultSet;

    struct KernelResult {
        std::string         mEntryName;
        bool				mSkipped			= true;
        bool				mCompiledCorrectly	= false;
        unsigned int        mIterations         = 0;
        std::string     	mExceptionString;
        InvocationResultSet mInvocations;
    };

    typedef std::vector<KernelResult> KernelResultSet;

    struct ModuleResult {
        std::string     mModuleName;
        std::string     mExceptionString;
        bool            mLoadedCorrectly    = false;
        KernelResultSet mKernels;
    };

    typedef std::vector<ModuleResult> ModuleResultSet;

    typedef InvocationResult (test_fn_signature)(clspv_utils::kernel&             kernel,
                                                 const std::vector<std::string>&  args,
                                                 bool                             verbose);

    typedef std::function<test_fn_signature> test_kernel_fn;

    typedef std::vector<test_kernel_fn> test_kernel_series;

    struct kernel_test_map {
        std::string                 entry;
        test_kernel_series          tests;
        vk::Extent3D                workgroupSize;
        std::vector<std::string>    args;
        unsigned int                iterations      = 1;
        bool                        verbose         = false;
    };

    struct module_test_bundle {
        std::string                     name;
        std::vector<kernel_test_map>    kernelTests;
    };

    template<typename T>
    bool pixel_compare(const T &l, const T &r) {
        return details::pixel_comparator<T>::is_equal(l, r);
    }

    template <typename PixelType, typename Iterator>
    void invert_pixel_buffer(Iterator first, Iterator last) {
        std::transform(first, last, first, [](const PixelType& p) {
            gpu_types::float4 p_inverted = pixels::traits<gpu_types::float4>::translate(p);

            p_inverted.x = std::fmod(p_inverted.x + 0.3f, 1.0f);
            p_inverted.y = std::fmod(p_inverted.y + 0.3f, 1.0f);
            p_inverted.z = std::fmod(p_inverted.z + 0.3f, 1.0f);
            p_inverted.w = std::fmod(p_inverted.w + 0.3f, 1.0f);

            return pixels::traits<PixelType>::translate(p_inverted);
        });
    }

    template <typename SrcPixelType, typename DstPixelType, typename SrcIterator, typename DstIterator>
    void copy_pixel_buffer(SrcIterator first, SrcIterator last, DstIterator dst) {
        std::transform(first, last, dst, [](const SrcPixelType& p) {
            return pixels::traits<DstPixelType>::translate(p);
        });
    }

    template <typename PixelType, typename OutputIterator>
    void fill_random_pixels(OutputIterator first, OutputIterator last) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, nextafterf(1.0f, std::numeric_limits<float>::max()));

        std::generate(first, last, [&dis,&gen]() {
            return pixels::traits<PixelType>::translate((gpu_types::float4){ dis(gen), dis(gen), dis(gen), dis(gen) });
        });
    }

    template<typename ExpectedPixelType, typename ObservedPixelType>
    void check_result(ExpectedPixelType expected_pixel,
                      ObservedPixelType observed_pixel,
                      vk::Extent3D      coord,
                      bool              verbose,
                      InvocationResult& result) {
        typedef typename details::pixel_promotion<ExpectedPixelType, ObservedPixelType>::promotion_type promotion_type;

        auto expected = pixels::traits<promotion_type>::translate(expected_pixel);
        auto observed = pixels::traits<promotion_type>::translate(observed_pixel);

        const bool pixel_is_correct = pixel_compare(observed, expected);
        if (pixel_is_correct) {
            ++result.mNumCorrect;
        }
        else {
            ++result.mNumErrors;
            if (verbose) {
                const std::string expectedString = pixels::traits<decltype(expected_pixel)>::toString(
                        expected_pixel);
                const std::string observedString = pixels::traits<decltype(observed_pixel)>::toString(
                        observed_pixel);

                const std::string expectedPromotionString = pixels::traits<decltype(expected)>::toString(
                        expected);
                const std::string observedPromotionString = pixels::traits<decltype(observed)>::toString(
                        observed);

                std::ostringstream os;
                os << (pixel_is_correct ? "CORRECT  " : "INCORRECT")
                   << ": pixel{x:" << coord.width << ", y:" << coord.height << ", z:" << coord.depth << "}"
                   << " expected:" << expectedString << " observed:" << observedString
                   << " expectedPromotion:" << expectedPromotionString << " observedPromotion:"
                   << observedPromotionString;
                result.mMessages.push_back(os.str());
            }
        }
    }

    template<typename ObservedPixelType, typename ExpectedPixelType>
    void check_results(const ObservedPixelType* observed_pixels,
                       vk::Extent3D             extent,
                       int                      pitch,
                       ExpectedPixelType        expected,
                       bool                     verbose,
                       InvocationResult&        result) {
        auto row = observed_pixels;
        for (vk::Extent3D coord; coord.depth < extent.depth; ++coord.depth) {
            for (coord.height = 0; coord.height < extent.height; ++coord.height, row += pitch) {
                auto p = row;
                for (coord.width = 0; coord.width < extent.width; ++coord.width, ++p) {
                    check_result(expected, *p, coord, verbose, result);
                }
            }
        }
    }

    template<typename ExpectedPixelType, typename ObservedPixelType>
    void check_results(const ExpectedPixelType* expected_pixels,
                       const ObservedPixelType* observed_pixels,
                       vk::Extent3D             extent,
                       int                      pitch,
                       bool                     verbose,
                       InvocationResult&        result) {
        auto expected_row = expected_pixels;
        auto observed_row = observed_pixels;
        for (vk::Extent3D coord; coord.depth < extent.depth; ++coord.depth) {
            for (coord.height = 0; coord.height < extent.height; ++coord.height, expected_row += pitch, observed_row += pitch) {
                auto expected_p = expected_row;
                auto observed_p = observed_row;
                for (coord.width = 0; coord.width < extent.width; ++coord.width, ++expected_p, ++observed_p) {
                    check_result(*expected_p, *observed_p, coord, verbose, result);
                }
            }
        }
    }

    KernelResult test_kernel(clspv_utils::kernel_module&        module,
                             const kernel_test_map&             kernelTest);

    ModuleResult test_module(clspv_utils::device_t&                 device,
                             const std::string&                     moduleName,
                             const std::vector<kernel_test_map>&    kernelTests);
    
}

#endif //CLSPVTEST_TEST_UTILS_HPP
