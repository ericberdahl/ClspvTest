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
    if (0 <= index && index < (sizeof(kTwoPowX) / sizeof(kTwoPowX[0]))) {
        return kTwoPowX[index];
    }

    return -1.0f;
}

inline float GetStructValue(int index) {
    switch (index) {
        case 0:
            return kTwoPowX_struct.m0;
        case 1:
            return kTwoPowX_struct.m1;
        case 2:
            return kTwoPowX_struct.m2;
        case 3:
            return kTwoPowX_struct.m3;
        case 4:
            return kTwoPowX_struct.m4;
        case 5:
            return kTwoPowX_struct.m5;
        case 6:
            return kTwoPowX_struct.m6;
        case 7:
            return kTwoPowX_struct.m7;
        case 8:
            return kTwoPowX_struct.m8;
        case 9:
            return kTwoPowX_struct.m9;
        default:
            return -1.0f;
    }
}

__kernel void ReadConstantArray(__global float* outDest, int inWidth)
{
    int index = KernelX();
    outDest[index] = GetArrayValue(index);
}


__kernel void ReadConstantStruct(__global float* outDest, int inWidth)
{
    int index = KernelX();
    outDest[index] = GetStructValue(index);
}

