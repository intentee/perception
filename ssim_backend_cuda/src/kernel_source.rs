pub(crate) const KERNEL_SOURCE: &str = r#"
__device__ __forceinline__ float srgb_eotf(float x) {
    if (x <= 0.04045f) {
        return x / 12.92f;
    }
    return powf((x + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float lab_f(float c) {
    if (c > 0.008856451679035631f) {
        return cbrtf(c);
    }
    return c * (841.0f / 108.0f) + (4.0f / 29.0f);
}

__device__ __forceinline__ void linear_to_lab(
    float r, float g, float b, float* out_l, float* out_a, float* out_b
) {
    float x = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
    float y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
    float z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;
    float fx = lab_f(x / 0.95047f);
    float fy = lab_f(y);
    float fz = lab_f(z / 1.08883f);
    float l = 116.0f * fy - 16.0f;
    float a = 500.0f * (fx - fy);
    float bb = 200.0f * (fy - fz);
    *out_l = l / 100.0f;
    *out_a = a / 256.0f + 0.5f;
    *out_b = bb / 256.0f + 0.5f;
}

extern "C" __global__ void srgb_gray_to_linear(const float* srgb, float* linear, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    linear[i] = srgb_eotf(srgb[i]);
}

extern "C" __global__ void srgb_rgb_to_linear(
    const float* interleaved, float* r, float* g, float* b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    r[i] = srgb_eotf(interleaved[3 * i + 0]);
    g[i] = srgb_eotf(interleaved[3 * i + 1]);
    b[i] = srgb_eotf(interleaved[3 * i + 2]);
}

extern "C" __global__ void srgba_to_linear_premult(
    const float* interleaved, float* r, float* g, float* b, float* a, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float alpha = interleaved[4 * i + 3];
    r[i] = srgb_eotf(interleaved[4 * i + 0]) * alpha;
    g[i] = srgb_eotf(interleaved[4 * i + 1]) * alpha;
    b[i] = srgb_eotf(interleaved[4 * i + 2]) * alpha;
    a[i] = alpha;
}

extern "C" __global__ void linear_gray_to_lab(const float* linear, float* lab, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float l, a, b;
    float value = linear[i];
    linear_to_lab(value, value, value, &l, &a, &b);
    lab[i] = l;
}

extern "C" __global__ void linear_rgb_to_lab(
    const float* r, const float* g, const float* b,
    float* out_l, float* out_a, float* out_b, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    linear_to_lab(r[i], g[i], b[i], &out_l[i], &out_a[i], &out_b[i]);
}

extern "C" __global__ void linear_rgba_to_lab(
    const float* r, const float* g, const float* b, const float* a,
    float* out_l, float* out_a, float* out_b, int width, int height
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    int x = i % width;
    int y = i / width;
    float red = r[i];
    float green = g[i];
    float blue = b[i];
    float alpha = a[i];
    if (alpha < 1.0f) {
        float uncovered = 1.0f - alpha;
        int dither = (x + 11) ^ (y + 11);
        if (dither & 16) red += uncovered;
        if (dither & 8) green += uncovered;
        if (dither & 32) blue += uncovered;
    }
    linear_to_lab(red, green, blue, &out_l[i], &out_a[i], &out_b[i]);
}

extern "C" __global__ void blur_horizontal(
    const float* src, float* dst, const float* k, int width, int height
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    int x = i % width;
    int base = (i / width) * width;
    int l1 = max(x - 1, 0), r1 = min(x + 1, width - 1);
    int l2 = max(x - 2, 0), r2 = min(x + 2, width - 1);
    int l3 = max(x - 3, 0), r3 = min(x + 3, width - 1);
    float acc = src[base + x] * k[3];
    acc = fmaf(src[base + l3] + src[base + r3], k[0], acc);
    acc = fmaf(src[base + l2] + src[base + r2], k[1], acc);
    acc = fmaf(src[base + l1] + src[base + r1], k[2], acc);
    dst[i] = acc;
}

extern "C" __global__ void blur_vertical(
    const float* src, float* dst, const float* k, int width, int height
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    int x = i % width;
    int y = i / width;
    int t1 = max(y - 1, 0), b1 = min(y + 1, height - 1);
    int t2 = max(y - 2, 0), b2 = min(y + 2, height - 1);
    int t3 = max(y - 3, 0), b3 = min(y + 3, height - 1);
    float acc = src[y * width + x] * k[3];
    acc = fmaf(src[t3 * width + x] + src[b3 * width + x], k[0], acc);
    acc = fmaf(src[t2 * width + x] + src[b2 * width + x], k[1], acc);
    acc = fmaf(src[t1 * width + x] + src[b1 * width + x], k[2], acc);
    dst[i] = acc;
}

extern "C" __global__ void multiply(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * b[i];
}

__device__ __forceinline__ float triangle_weight(float t) {
    float at = fabsf(t);
    return at < 1.0f ? 1.0f - at : 0.0f;
}

extern "C" __global__ void downsample_vertical(
    const float* src, float* dst, int width, int src_h, int dst_h
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * dst_h) return;
    int x = i % width;
    int outy = i / width;
    float ratio = (float)src_h / (float)dst_h;
    float sratio = ratio < 1.0f ? 1.0f : ratio;
    float support = sratio;
    float inputy = ((float)outy + 0.5f) * ratio;
    int top = (int)floorf(inputy - support);
    if (top < 0) top = 0;
    if (top > src_h - 1) top = src_h - 1;
    int bottom = (int)ceilf(inputy + support);
    if (bottom < top + 1) bottom = top + 1;
    if (bottom > src_h) bottom = src_h;
    inputy -= 0.5f;
    float sum = 0.0f;
    for (int yy = top; yy < bottom; yy++) {
        sum += triangle_weight(((float)yy - inputy) / sratio);
    }
    float acc = 0.0f;
    for (int yy = top; yy < bottom; yy++) {
        float w = triangle_weight(((float)yy - inputy) / sratio) / sum;
        acc += w * src[yy * width + x];
    }
    if (acc < 0.0f) acc = 0.0f;
    if (acc > 1.0f) acc = 1.0f;
    dst[i] = acc;
}

extern "C" __global__ void downsample_horizontal(
    const float* src, float* dst, int src_w, int dst_w, int height
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dst_w * height) return;
    int outx = i % dst_w;
    int y = i / dst_w;
    float ratio = (float)src_w / (float)dst_w;
    float sratio = ratio < 1.0f ? 1.0f : ratio;
    float support = sratio;
    float inputx = ((float)outx + 0.5f) * ratio;
    int left = (int)floorf(inputx - support);
    if (left < 0) left = 0;
    if (left > src_w - 1) left = src_w - 1;
    int right = (int)ceilf(inputx + support);
    if (right < left + 1) right = left + 1;
    if (right > src_w) right = src_w;
    inputx -= 0.5f;
    float sum = 0.0f;
    for (int xx = left; xx < right; xx++) {
        sum += triangle_weight(((float)xx - inputx) / sratio);
    }
    float acc = 0.0f;
    for (int xx = left; xx < right; xx++) {
        float w = triangle_weight(((float)xx - inputx) / sratio) / sum;
        acc += w * src[y * src_w + xx];
    }
    if (acc < 0.0f) acc = 0.0f;
    if (acc > 1.0f) acc = 1.0f;
    dst[i] = acc;
}

extern "C" __global__ void ssim_accumulate(
    const float* mu1, const float* mu2,
    const float* sq1, const float* sq2, const float* cross,
    float* a_mu1_sq, float* a_mu2_sq, float* a_mu1_mu2,
    float* a_sigma1, float* a_sigma2, float* a_sigma12, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float m1 = mu1[i];
    float m2 = mu2[i];
    float m1m1 = m1 * m1;
    float m2m2 = m2 * m2;
    float m1m2 = m1 * m2;
    a_mu1_sq[i] += m1m1;
    a_mu2_sq[i] += m2m2;
    a_mu1_mu2[i] += m1m2;
    a_sigma1[i] += sq1[i] - m1m1;
    a_sigma2[i] += sq2[i] - m2m2;
    a_sigma12[i] += cross[i] - m1m2;
}

extern "C" __global__ void ssim_finalize(
    const float* a_mu1_sq, const float* a_mu2_sq, const float* a_mu1_mu2,
    const float* a_sigma1, const float* a_sigma2, const float* a_sigma12,
    int channel_count, int n, float* out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float inv = 1.0f / (float)channel_count;
    float mu1_sq = a_mu1_sq[i] * inv;
    float mu2_sq = a_mu2_sq[i] * inv;
    float mu1_mu2 = a_mu1_mu2[i] * inv;
    float sigma1 = a_sigma1[i] * inv;
    float sigma2 = a_sigma2[i] * inv;
    float sigma12 = a_sigma12[i] * inv;
    const float c1 = 0.01f * 0.01f;
    const float c2 = 0.03f * 0.03f;
    float numerator = fmaf(2.0f, mu1_mu2, c1) * fmaf(2.0f, sigma12, c2);
    float denominator = (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2);
    out[i] = numerator / denominator;
}

extern "C" __global__ void reduce_sum(const float* data, double* partials, int n) {
    extern __shared__ double shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (i < n) ? (double)data[i] : 0.0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = shared[0];
}

extern "C" __global__ void reduce_abs_deviation(
    const float* data, double adjusted_mean, double* partials, int n
) {
    extern __shared__ double shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = (i < n) ? fabs(adjusted_mean - (double)data[i]) : 0.0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = shared[0];
}
"#;
