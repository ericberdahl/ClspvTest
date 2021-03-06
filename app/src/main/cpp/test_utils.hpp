//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_TEST_UTILS_HPP
#define CLSPVTEST_TEST_UTILS_HPP

#include "clspv_utils/invocation.hpp"
#include "clspv_utils/kernel.hpp"
#include "fp_utils.hpp"
#include "gpu_types.hpp"
#include "pixels.hpp"

#include <vulkan/vulkan.hpp>

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

    class StopWatch
    {
    public:
        typedef std::chrono::high_resolution_clock  clock;
        typedef std::chrono::duration<double>       duration;

        StopWatch();

        void        restart();
        duration    getSplitTime() const;

    private:
        clock::time_point   mStartTime;
    };

    struct Evaluation {
        bool                            mSkipped    = false;
        unsigned int                    mNumCorrect = 0;
        unsigned int                    mNumErrors  = 0;
        std::vector<std::string>        mMessages;

        Evaluation& operator+=(const Evaluation& other);
    };

    struct InvocationResult {
        InvocationResult() : mEvalTime(0.0) {}

        std::string                     mParameters;
        clspv_utils::execution_time_t   mExecutionTime;
        Evaluation                      mEvaluation;
        std::chrono::duration<double>   mEvalTime;
    };

    struct InvocationTest {
        typedef std::pair<const InvocationTest*,InvocationResult>   result;

        typedef InvocationResult (test_fn_signature)(clspv_utils::kernel&             kernel,
                                                     const std::vector<std::string>&  args,
                                                     bool                             verbose);

        typedef std::function<test_fn_signature> test_fn;

        typedef std::vector<InvocationResult> (time_fn_signature)(
                                                     clspv_utils::kernel&             kernel,
                                                     const std::vector<std::string>&  args,
                                                     unsigned int                     iterations,
                                                     bool                             verbose);

        typedef std::function<time_fn_signature> time_fn;

        std::string mVariation;
        test_fn     mTestFn;
        time_fn     mTimeFn;
    };

    struct KernelResult {
        typedef std::vector<InvocationTest::result> results;

        bool			mSkipped			= true;
        bool			mCompiledCorrectly	= false;
        std::string     mExceptionString;
        results         mInvocationResults;
    };

    struct KernelTest {
        typedef std::pair<const KernelTest*,KernelResult>   result;
        typedef std::vector<InvocationTest>                 invocation_tests;
        typedef std::vector<std::string>                    test_arguments;

        std::string         mEntryName;
        vk::Extent3D        mWorkgroupSize;
        test_arguments      mArguments;
        unsigned int        mTimingIterations   = 0;
        bool                mIsVerbose          = false;
        invocation_tests    mInvocationTests;
    };

    struct ModuleResult {
        typedef std::vector<KernelTest::result> results;

        std::string                 mExceptionString;
        bool                        mLoadedCorrectly    = false;
        std::vector<std::string>    mUntestedEntryPoints;
        results                     mKernelResults;
    };

    struct ModuleTest {
        typedef std::pair<const ModuleTest*,ModuleResult>   result;
        typedef std::vector<KernelTest>                     kernel_tests;

        std::string     mName;
        kernel_tests    mKernelTests;
    };

    class Test {
    public:
                Test();
        virtual ~Test();

        virtual std::string getParameterString() const;
        virtual void        prepare();
        virtual clspv_utils::execution_time_t   run(clspv_utils::kernel& kernel) = 0;
        virtual Evaluation  evaluate(bool verbose);
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
    Evaluation evaluate_result(ExpectedPixelType expected_pixel,
                               ObservedPixelType observed_pixel,
                               vk::Extent3D      coord,
                               bool              verbose) {
        Evaluation result;

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

        return result;
    }

    template<typename ObservedPixelType, typename ExpectedPixelType>
    Evaluation check_results(const ObservedPixelType* observed_pixels,
                             vk::Extent3D             extent,
                             int                      pitch,
                             ExpectedPixelType        expected,
                             bool                     verbose) {
        Evaluation result;

        auto row = observed_pixels;
        for (vk::Extent3D coord; coord.depth < extent.depth; ++coord.depth) {
            for (coord.height = 0; coord.height < extent.height; ++coord.height, row += pitch) {
                auto p = row;
                for (coord.width = 0; coord.width < extent.width; ++coord.width, ++p) {
                    result += evaluate_result(expected, *p, coord, verbose);
                }
            }
        }

        return result;
    }

    template<typename ExpectedPixelType, typename ObservedPixelType>
    Evaluation check_results(const ExpectedPixelType* expected_pixels,
                             const ObservedPixelType* observed_pixels,
                             vk::Extent3D             extent,
                             int                      pitch,
                             bool                     verbose) {
        Evaluation result;

        auto expected_row = expected_pixels;
        auto observed_row = observed_pixels;
        for (vk::Extent3D coord; coord.depth < extent.depth; ++coord.depth) {
            for (coord.height = 0; coord.height < extent.height; ++coord.height, expected_row += pitch, observed_row += pitch) {
                auto expected_p = expected_row;
                auto observed_p = observed_row;
                for (coord.width = 0; coord.width < extent.width; ++coord.width, ++expected_p, ++observed_p) {
                    result += evaluate_result(*expected_p, *observed_p, coord, verbose);
                }
            }
        }

        return result;
    }

    InvocationResult run_test(clspv_utils::kernel&              kernel,
                              const std::vector<std::string>&   args,
                              bool                              verbose,
                              Test&                             test);

    std::vector<InvocationResult> time_test(clspv_utils::kernel&             kernel,
                                            const std::vector<std::string>&  args,
                                            unsigned int                     iterations,
                                            bool                             verbose,
                                            Test&                            test);

    template <typename Test>
    InvocationResult run_test(clspv_utils::kernel&              kernel,
                              const std::vector<std::string>&   args,
                              bool                              verbose)
    {
        InvocationResult result;

        try
        {
            Test test(kernel, args);
            result = run_test(kernel, args, verbose, test);
        }
        catch(const std::exception& e)
        {
            result.mEvaluation.mSkipped = true;
            result.mEvaluation.mMessages.push_back(e.what());
        }
        catch(...)
        {
            result.mEvaluation.mSkipped = true;
            result.mEvaluation.mMessages.push_back("Unknown exception running test");
        }

        return result;
    }

    template <typename Test>
    std::vector<InvocationResult> time_test(clspv_utils::kernel&             kernel,
                                            const std::vector<std::string>&  args,
                                            unsigned int                     iterations,
                                            bool                             verbose)
    {
        Test test(kernel, args);
        return time_test(kernel, args, iterations, verbose, test);
    }

    template <typename Test>
    InvocationTest make_invocation_test(std::string variation)
    {
        return InvocationTest{ variation, run_test<Test>, time_test<Test> };
    }

    KernelTest::result test_kernel(clspv_utils::module& module,
                                   const KernelTest&    kernelTest);

    ModuleTest::result test_module(clspv_utils::device& inDevice,
                                   const ModuleTest&    moduleTest);

    InvocationTest createNullInvocationTest();
    
}

#endif //CLSPVTEST_TEST_UTILS_HPP
