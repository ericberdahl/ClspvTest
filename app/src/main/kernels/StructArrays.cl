#define NUM_ARRAY_ELEMENTS 18

typedef struct {
    float   arr[NUM_ARRAY_ELEMENTS];
} FloatArrayWrapper;

void FillOneElement(__global FloatArrayWrapper* outWrapper,
                    unsigned int wrapperIndex,
                    unsigned int elementIndex)
{
    __global FloatArrayWrapper* wrapper = &outWrapper[wrapperIndex];

    wrapper->arr[elementIndex] = sizeof(FloatArrayWrapper) * 10000.0f
                                 + wrapperIndex * 100.0f
                                 + (float) elementIndex;
}

__kernel void FillStructArray(__global FloatArrayWrapper* outWrapper)
{
    unsigned int x = get_global_id(0);

    for (unsigned int i = 0; i < NUM_ARRAY_ELEMENTS; ++i)
    {
        FillOneElement(outWrapper, x, i);
    }
}
