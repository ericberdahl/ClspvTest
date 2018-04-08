# ClspvTest

One example of a Vulkan application that runs OpenCL C kernels, compiled to SPIR-V wwth [clspv][clspv], in a Vulkan compute pipeline.

## Requirements

To build ClspvTest, you will need:

* [clspv][clspv]
* [glslangValidator][glslang]
* [Android Studio][android-studio]

CMakeLists.txt expects to find:

* clspv at `/usr/local/bin/clspv`
* glslangValidator at `/usr/local/bin/glslangValidator`
* spirv-opt at `/usr/local/bin/spirv-opt` (I recommend using the version of spirv-opt built along with your clspv distribution)

## Build instructions

1. Run Android Studio
2. Select `Open an existing Android Studio project`
3. Navigate to the location at which you cloned this repo
4. Build and/or run the project.

## Running ClspvTest

ClspvTest runs on Android. You will need an Android device on which to run the application because the emulators available in Android Studio do not support Vulkan.

ClspvTest provides no UI on the Android device. All output is represented in messages written to the Android log (e.g. visible via logcat).

[android-studio]: https://developer.android.com/studio/index.html
[clspv]: https://github.com/google/clspv
[glslang]: https://github.com/KhronosGroup/glslang
