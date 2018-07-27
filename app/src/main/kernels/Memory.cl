float saturate(float inX)
{
    return clamp(inX, 0.0f, 1.0f);
}

int KernelX()
{
    return get_global_id(0);
}

int KernelY()
{
    return get_global_id(1);
}

int KernelZ()
{
    return get_global_id(2);
}

unsigned int ReadUCharIndex(
    __global const int* inSource,
    int                 inIndex)
{
    __global const uint* inPtr = ((__global const uint*)inSource) + inIndex/4;

    const int shift = 8 * (inIndex % 4);
    return ((*inPtr >> shift) & 0xFF);
}

void WriteUCharIndex(
    __global int*   outDest,
    int             inIndex,
    unsigned int    inValue)
{
    __global uint* outPtr = ((__global uint*)outDest) + inIndex/4;

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
        int srcIndex = mul24(y, inSrcPitch) + x;

        if (inSrcChannelType == CLK_UNORM_INT8)
        {
            if (inSrcChannelOrder == CLK_R)
            {
                uint input = ReadUCharIndex(((__global const uchar*)inSrc) + inSrcOffset/sizeof(const uchar), srcIndex);
                pixel.x = input / 255.0f;
            }
            else
            {
                uchar4 input = (((__global const uchar4*)inSrc) + inSrcOffset/sizeof(const uchar4))[srcIndex];
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
                pixel.x = vload_half(srcIndex, ((__global const half*)inSrc) + inSrcOffset/sizeof(const half));
            }
            else
            {
                pixel = vload_half4(srcIndex, ((__global const half*)inSrc) + inSrcOffset/sizeof(const half));
            }
        }
        else
        {
            if (inSrcChannelOrder == CLK_R)
            {
                float input = (((__global const float*)inSrc) + inSrcOffset/sizeof(const float))[srcIndex];
                pixel.x = input;
            }
            else if (inSrcChannelOrder == CLK_RG)
            {
                float2 input = (((__global const float2*)inSrc) + inSrcOffset/sizeof(const float2))[srcIndex];
                pixel.x = input.x;
                pixel.y = input.y;
            }
            else
            {
                pixel = (((__global const float4*)inSrc) + inSrcOffset/sizeof(const float4))[srcIndex];
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
        float4 pixel = read_imagef(inImage, copyImageToBufferSampler, (float2)(x, y));
        int dstIndex = mul24(y, inDestPitch) + x;

        if (inSwapComponents)
        {
            pixel = (float4)(pixel.z, pixel.y, pixel.x, pixel.w);
        }

        if (inDestChannelType == CLK_UNORM_INT8)
        {
            if (inDestChannelOrder == CLK_R)
            {
                uint output = pixel.x * 255.0 + 0.5f;
                WriteUCharIndex(((__global uchar*)outDest) + inDestOffset/sizeof(uchar), dstIndex, output);
            }
            else
            {
                uchar4 output = (uchar4)(pixel.x * 255.0f + 0.5f, pixel.y * 255.0f + 0.5f, pixel.z * 255.0f + 0.5f, pixel.w * 255.0f + 0.5f);
                (((__global uchar4*)outDest) + inDestOffset/sizeof(uchar4))[dstIndex] = output;
            }
        }
        else if (inDestChannelType == CLK_HALF_FLOAT)
        {
            if (inDestChannelOrder == CLK_R)
            {
                vstore_half_rtz(pixel.x, dstIndex, ((__global half*)outDest) + inDestOffset/sizeof(half));
            }
            else
            {
                vstore_half4_rtz(pixel, dstIndex, ((__global half*)outDest) + inDestOffset/sizeof(half));
            }
        }
        else
        {
            if (inDestChannelOrder == CLK_R)
            {
                (((__global float*)outDest) + inDestOffset/sizeof(float))[dstIndex] = pixel.x;
            }
            else if (inDestChannelOrder == CLK_RG)
            {
                (((__global float2*)outDest) + inDestOffset/sizeof(float2))[dstIndex] = (float2)(pixel.x, pixel.y);
            }
            else
            {
                (((__global float4*)outDest) + inDestOffset/sizeof(float4))[dstIndex] = pixel;
            }
        }
    }
}


const sampler_t linearSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void Resample2DImage(
    __read_only image2d_t   inImage,
    __global float4*        outDest,
    int                     inDestWidth,
    int                     inDestHeight)
{
    int x = KernelX();
    int y = KernelY();

    if (x < inDestWidth && y < inDestHeight)
    {
        float2 srcCoord = (float2)( ((float)x + 0.5f)/((float)inDestWidth),
                                    ((float)y + 0.5f)/((float)inDestHeight) );
        int destIndex = (y * inDestWidth) + x;

        float4 pixel = read_imagef(inImage, linearSampler, srcCoord);

        outDest[destIndex] = pixel;
    }
}

__kernel void Resample3DImage(
    __read_only image3d_t   inImage,
    __global float4*        outDest,
    int                     inDestWidth,
    int                     inDestHeight,
    int                     inDestDepth)
{
    int x = KernelX();
    int y = KernelY();
    int z = KernelZ();

    if (x < inDestWidth && y < inDestHeight && z < inDestDepth)
    {
        float4 srcCoord = (float4)( ((float)x + 0.5f)/((float)inDestWidth),
                                    ((float)y + 0.5f)/((float)inDestHeight),
                                    ((float)z + 0.5f)/((float)inDestDepth),
                                    0.0f );
        int destIndex = (((z * inDestHeight) + y) * inDestWidth) + x;

        float4 pixel = read_imagef(inImage, linearSampler, srcCoord);

        outDest[destIndex] = pixel;
    }
}

