typedef struct __attribute__ ((aligned (16)))
{
    float b, g, r, a;
} PixelRGB;

float saturate(float inX)
{
    return clamp(inX, 0.0f, 1.0f);
}

uint2 KernelXYUnsigned()
{
    return (uint2)(get_global_id(0), get_global_id(1));
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

float4 ConvertRGB_To_Components(PixelRGB inPixel)
{
    return (float4)(inPixel.b, inPixel.g, inPixel.r, inPixel.a);
}

PixelRGB ConvertComponents_To_RGB(float4 inPixel)
{
    PixelRGB val = { inPixel.x, inPixel.y, inPixel.z, inPixel.w };
    return val;
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

float4 ReadFloat4(
    const __global float4*  inImage,
    int                     inIndex,
    bool                    is16Bit)
{
    if (is16Bit)
    {
        return vload_half4(inIndex, (const __global half*)inImage);
    }
    return inImage[inIndex];
}

void WriteFloat4(
    float4              inPixel,
    __global float4*    outImage,
    int                 inIndex,
    bool                is16Bit)
{
    if (is16Bit)
    {
        vstorea_half4_rtz(inPixel, inIndex, (__global half*)outImage);
    }
    else
    {
        outImage[inIndex] = inPixel;
    }
}

float4 ReadPixelIndex(
    __global float4 const*  inImage,
    bool                    is16Bit,
    int                     inIndex)
{
    return ReadFloat4(inImage, inIndex, is16Bit);
}

void WritePixelIndex(
    float4              inPixel,
    __global float4*    outImage,
    bool                is16Bit,
    int                 inIndex)
{
    WriteFloat4(inPixel, outImage, inIndex, is16Bit);
}

float4 ReadPixel(
    const __global float4*  inImage,
    int                     inPitch,
    bool                    is16Bit,
    int                     inX,
    int                     inY)
{
    return ReadPixelIndex(inImage, is16Bit, mul24(inY, inPitch) + inX);
}

void WritePixel(
    float4              inPixel,
    __global float4*    outImage,
    int                 inPitch,
    bool                is16Bit,
    int                 inX,
    int                 inY)
{
    WritePixelIndex(inPixel, outImage, is16Bit, mul24(inY, inPitch) + inX);
}

PixelRGB ReadRGBPixel(
    const __global float4*  inImage,
    int                     inPitch,
    bool                    is16Bit,
    int                     inX,
    int                     inY)
{
    return ConvertComponents_To_RGB(ReadPixel(inImage, inPitch, is16Bit, inX, inY));
}

void WriteRGBPixel(
    PixelRGB            inPixel,
    __global float4*    outImage,
    int                 inPitch,
    bool                is16Bit,
    int                 inX,
    int                 inY)
{
    WritePixel(ConvertRGB_To_Components(inPixel), outImage, inPitch, is16Bit, inX, inY);
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
                vstorea_half4_rtz(pixel, dstIndex, ((__global half*)outDest) + inDestOffset/sizeof(half));
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

__kernel void CopyBufferToBufferKernel(const __global float4* inSrc,
                                       __global float4*       outDest,
                                       int                    inSrcPitch,
                                       int                    inSrcOffset,
                                       int                    inDestPitch,
                                       int                    inDestOffset,
                                       int                    inIs32BitBuffer,
                                       int                    inWidth,
                                       int                    inHeight)
{
    uint2 inXY = KernelXYUnsigned();

    const __global float4*  pSrc  = inSrc   + inSrcOffset;
    __global float4*        pDest = outDest + inDestOffset;

    if ((int)inXY.x < inWidth && (int)inXY.y < inHeight)
    {
        WriteRGBPixel(
            ReadRGBPixel(pSrc, inSrcPitch, !inIs32BitBuffer, inXY.x, inXY.y),
            pDest,
            inDestPitch,
            !inIs32BitBuffer,
            inXY.x,
            inXY.y);
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

