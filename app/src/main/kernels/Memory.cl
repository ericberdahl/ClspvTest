static float saturate(float inX)
{
    return clamp(inX, 0.0f, 1.0f);
}

static int KernelX()
{
    return get_global_id(0);
}

static int KernelY()
{
    return get_global_id(1);
}


static unsigned int ReadUCharIndex(
    __global const int* inSource,
    int                 inIndex)
{
    __global const uint* inPtr = (((__global const uint*)inSource) + (0)/sizeof(const uint)) + inIndex/4;

    const int shift = 8 * (inIndex % 4);
    return ((*inPtr >> shift) & 0xFF);
}

static void WriteUCharIndex(
    __global int*   outDest,
    int             inIndex,
    unsigned int    inValue)
{
    __global uint* outPtr = (((__global uint*)outDest) + (0)/sizeof(uint)) + inIndex/4;

    const int shift = 8 * (inIndex % 4);
    const uint mask = (*outPtr & (0x000000FF << shift)) ^ (inValue << shift);
    atomic_xor(outPtr, mask);
}

__kernel void CopyBufferToImageKernel(
    __global const float4*  inSrc,
    __write_only image2d_t  outImage,
    int                     inSrcOffset,
    int                     inSrcPitch,
    int                     inSrcChannelOrder,
    int                     inSrcChannelType,
    int                     inSwapComponents,
    int                     inPremultiply,
    int                     inWidth,
    int                     inHeight)
{
    int x = KernelX();
    int y = KernelY();

    if (x < inWidth && y < inHeight)
    {
        float4 pixel;
        if (inSrcChannelType == CLK_UNORM_INT8)
        {
            if (inSrcChannelOrder == CLK_R)
            {
                uint input = ReadUCharIndex((((__global const uchar*)inSrc) + (inSrcOffset)/sizeof(const uchar)), (mul24((y), (inSrcPitch)) + (x)));
                pixel.x = input / 255.0f;
            }
            else
            {
                uchar4 input = ((((__global const uchar4*)inSrc) + (inSrcOffset)/sizeof(const uchar4)))[mul24(((y)), ((inSrcPitch))) + (x)];
                pixel.x = input.x / 255.0f;
                pixel.y = input.y / 255.0f;
                pixel.z = input.z / 255.0f;
                pixel.w = input.w / 255.0f;
            }
        }
        else if (inSrcChannelType == CLK_HALF_FLOAT)
        {
            if (inSrcChannelOrder == CLK_R)
            {
                pixel.x = vload_half(mul24((y), (inSrcPitch)) + x, (((__global const half*)inSrc) + (inSrcOffset)/sizeof(const half)));
            }
            else
            {
                pixel = vload_half4(mul24((y), (inSrcPitch)) + x, (((__global const half*)inSrc) + (inSrcOffset)/sizeof(const half)));
            }
        }
        else
        {
            if (inSrcChannelOrder == CLK_R)
            {
                float input = ((((__global const float*)inSrc) + (inSrcOffset)/sizeof(const float)))[mul24(((y)), ((inSrcPitch))) + (x)];
                pixel.x = input;
            }
            else if (inSrcChannelOrder == CLK_RG)
            {
                float2 input = ((((__global const float2*)inSrc) + (inSrcOffset)/sizeof(const float2)))[mul24(((y)), ((inSrcPitch))) + (x)];
                pixel.x = input.x;
                pixel.y = input.y;
            }
            else
            {
                pixel = ((((__global const float4*)inSrc) + (inSrcOffset)/sizeof(const float4)))[mul24(((y)), ((inSrcPitch))) + (x)];
            }
        }

        if (inPremultiply)
        {
            pixel.x = pixel.x * saturate(pixel.w);
            pixel.y = pixel.y * saturate(pixel.w);
            pixel.z = pixel.z * saturate(pixel.w);
        }

        if (inSwapComponents)
        {
            pixel = (float4)(pixel.z, pixel.y, pixel.x, pixel.w);
        }

        write_imagef(outImage, (int2)(x, y), pixel);
    }
}

const sampler_t copyImageToBufferSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void CopyImageToBufferKernel(
    __read_only image2d_t   inImage,
    __global int*           outDest,
    int                     inDestOffset,
    int                     inDestPitch,
    int                     inDestChannelOrder,
    int                     inDestChannelType,
    int                     inSwapComponents,
    int                     inWidth,
    int                     inHeight)
{
    int x = KernelX();
    int y = KernelY();

    if (x < inWidth && y < inHeight)
    {
        float4 pixel = read_imagef(inImage, copyImageToBufferSampler, (int2)(x, y));

        if (inSwapComponents)
        {
            pixel = (float4)(pixel.z, pixel.y, pixel.x, pixel.w);
        }

        if (inDestChannelType == CLK_UNORM_INT8)
        {
            if (inDestChannelOrder == CLK_R)
            {
                uint output = pixel.x * 255.0 + 0.5f;
                WriteUCharIndex((((__global uchar*)outDest) + (inDestOffset)/sizeof(uchar)), (mul24((y), (inDestPitch)) + (x)), output);
            }
            else
            {
                uchar4 output = (uchar4)(pixel.x * 255.0f + 0.5f, pixel.y * 255.0f + 0.5f, pixel.z * 255.0f + 0.5f, pixel.w * 255.0f + 0.5f);
                ((((__global uchar4*)outDest) + (inDestOffset)/sizeof(uchar4)))[mul24((y), (inDestPitch)) + (x)] = (output);
            }
        }
        else if (inDestChannelType == CLK_HALF_FLOAT)
        {
            if (inDestChannelOrder == CLK_R)
            {
                vstore_half_rtz(pixel.x, mul24((y), (inDestPitch)) + x, (((__global half*)outDest) + (inDestOffset)/sizeof(half)));
            }
            else
            {
                vstore_half4_rtz(pixel, mul24((y), (inDestPitch)) + x, (((__global half*)outDest) + (inDestOffset)/sizeof(half)));
            }
        }
        else
        {
            if (inDestChannelOrder == CLK_R)
            {
                ((((__global float*)outDest) + (inDestOffset)/sizeof(float)))[mul24((y), (inDestPitch)) + (x)] = (pixel.x);
            }
            else if (inDestChannelOrder == CLK_RG)
            {
                ((((__global float2*)outDest) + (inDestOffset)/sizeof(float2)))[mul24((y), (inDestPitch)) + (x)] = ((float2)(pixel.x, pixel.y));
            }
            else
            {
                ((((__global float4*)outDest) + (inDestOffset)/sizeof(float4)))[mul24((y), (inDestPitch)) + (x)] = (pixel);
            }
        }
    }
}

