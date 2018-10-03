#define OUT_OF_BOUNDS_VALUE (-1.0f)

int KernelX()
{
    return get_global_id(0);
}

typedef struct {
    float   m0,
            m1,
            m2,
            m3,
            m4,
            m5,
            m6,
            m7,
            m8,
            m9,
            m10,
            m11;
} FloatStruct;

// Lookup table for the function 2^x, defined on the domain [0,9]
__constant float kTwoPowX[12] =
{
    1.0f,
    2.0f,
    4.0f,
    8.0f,
    16.0f,
    32.0f,
    64.0f,
    128.0f,
    256.0f,
    512.0f,
    1024.0f,
    2048.0f
};

__constant FloatStruct kTwoPowX_struct = {
    1.0f,
    2.0f,
    4.0f,
    8.0f,
    16.0f,
    32.0f,
    64.0f,
    128.0f,
    256.0f,
    512.0f,
    1024.0f,
    2048.0f
};

inline float GetArrayValue(int index) {
    float result = OUT_OF_BOUNDS_VALUE;

    if (0 <= index && index < (sizeof(kTwoPowX) / sizeof(kTwoPowX[0]))) {
        result = kTwoPowX[index];
    }

    return result;
}

inline float GetStructValue(int index) {
    if (index < 0)
        return OUT_OF_BOUNDS_VALUE;

    // This ugly way to turn an index into a member reference is necessary both because of a problem
    // initializing constant arrays in some devices, and bugs comparing against zero in those same
    // devices.
    float result = kTwoPowX_struct.m0;
    if (index > 0) {
        result = kTwoPowX_struct.m1;
        --index;
    }
    if (index > 0) {
        result = kTwoPowX_struct.m2;
        --index;
    }
    if (index > 0) {
        result = kTwoPowX_struct.m3;
        --index;
    }
    if (index > 0) {
        result = kTwoPowX_struct.m4;
        --index;
    }
    if (index > 0) {
        result = kTwoPowX_struct.m5;
        --index;
    }
    if (index > 0) {
        result = kTwoPowX_struct.m6;
        --index;
    }
    if (index > 0) {
        result = kTwoPowX_struct.m7;
        --index;
    }
    if (index > 0) {
        result = kTwoPowX_struct.m8;
        --index;
    }
    if (index > 0) {
        result = kTwoPowX_struct.m9;
        --index;
    }
    if (index > 0) {
        result = kTwoPowX_struct.m10;
        --index;
    }
    if (index > 0) {
        result = kTwoPowX_struct.m11;
        --index;
    }
    if (index > 0) {
        return OUT_OF_BOUNDS_VALUE;
    }

    return result;
}

void WriteFloat(__global float* inBuffer, int inIndex, float inValue) {
    inBuffer[inIndex] = inValue;
}

float ReadFloat(__constant float* inBuffer, int inIndex) {
    return inBuffer[inIndex];
}

void WriteFloatStruct(__global FloatStruct* inBuffer, int inIndex, FloatStruct inValue) {
    inBuffer[inIndex] = inValue;
}

FloatStruct ReadFloatStruct(__constant FloatStruct* inBuffer, int inIndex) {
    return inBuffer[inIndex];
}

__kernel void ReadConstantArray(__global float* outDest, int inWidth)
{
    int index = KernelX();
    if (index < inWidth) {
        outDest[index] = GetArrayValue(index);
    }
}

__kernel void ReadConstantStruct(__global float* outDest, int inWidth)
{
    int index = KernelX();
    if (index < inWidth) {
        outDest[index] = GetStructValue(index);
    }
}

__kernel void CopyConstantBufferArg(__constant float*   inSource,
                                    __global float*     outDest,
                                    int                 inWidth)
{
    int index = KernelX();
    if (index < inWidth) {
        WriteFloat(outDest, index, ReadFloat(inSource, index));
    }
}

__kernel void CopyConstantBufferStructArg(__constant FloatStruct*   inSource,
                                          __global FloatStruct*     outDest,
                                          int                       inWidth)
{
    int index = KernelX();
    if (index < inWidth) {
        WriteFloatStruct(outDest, index, ReadFloatStruct(inSource, index));
    }
}
