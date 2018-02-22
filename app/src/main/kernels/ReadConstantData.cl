#define OUT_OF_BOUNDS_VALUE (-1.0f)

int KernelX()
{
    return get_global_id(0);
}

struct Float10 {
    float   m0,
            m1,
            m2,
            m3,
            m4,
            m5,
            m6,
            m7,
            m8,
            m9;
};

// Lookup table for the function 2^x, defined on the domain [0,9]
__constant float kTwoPowX[10] =
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
    512.0f
};

__constant struct Float10 kTwoPowX_struct = {
    1.0f,
    2.0f,
    4.0f,
    8.0f,
    16.0f,
    32.0f,
    64.0f,
    128.0f,
    256.0f,
    512.0f
};

inline float GetArrayValue(int index) {
    float result = OUT_OF_BOUNDS_VALUE;

    if (0 <= index && index < (sizeof(kTwoPowX) / sizeof(kTwoPowX[0]))) {
        result = kTwoPowX[index];
    }

    return result;
}

inline float GetStructValue(int index) {
    float result = OUT_OF_BOUNDS_VALUE;

    switch (index) {
        case 0:
            result = kTwoPowX_struct.m0;
            break;
        case 1:
            result = kTwoPowX_struct.m1;
            break;
        case 2:
            result = kTwoPowX_struct.m2;
            break;
        case 3:
            result = kTwoPowX_struct.m3;
            break;
        case 4:
            result = kTwoPowX_struct.m4;
            break;
        case 5:
            result = kTwoPowX_struct.m5;
            break;
        case 6:
            result = kTwoPowX_struct.m6;
            break;
        case 7:
            result = kTwoPowX_struct.m7;
            break;
        case 8:
            result = kTwoPowX_struct.m8;
            break;
        case 9:
            result = kTwoPowX_struct.m9;
            break;
    }

    return result;
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

