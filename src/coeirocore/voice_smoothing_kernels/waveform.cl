// 固定2048点の周期窓を使い、文章長をコンパイル条件に含めない。
#define FFT_SIZE 2048
#define FFT_HALF 1024
#define LANES 256
#define SPLINE_R (-0.2679491924311227f)

DEVICE void finish_group(int start, int *used, int *count, GLOBAL int4 *groups, GLOBAL const float *marks) {
    if (*used - start >= 9) groups[(*count)++] = i4(start, *used-start, (int)ceil(marks[start]), (int)floor(marks[*used-1]));
    else *used = start;
}

KERNEL void collect_groups(GLOBAL const float *raster, int n,
                             GLOBAL const int2 *spans, int span_count, float sr,
                             GLOBAL float *marks, GLOBAL int4 *groups, GLOBAL int2 *state) {
    if (get_global_id(0)) return;
    int used = 0, count = 0;
    for (int s = 0; s < span_count; ++s) {
        int2 span = spans[s];
        int start = used;
        float previous = -1, previous_period = 0;
        for (int j = max(0, span.x); j < min(n, span.y); ++j) {
            float mark = raster[j];
            if (mark < span.x || mark >= span.y) continue;
            if (previous >= 0) {
                float period = mark-previous;
                int bad = period > sr/65 || period < sr/550;
                if (previous_period > 0) bad |= fmax(period/previous_period, previous_period/period) > 1.5f;
                if (bad) { finish_group(start, &used, &count, groups, marks); start = used; }
                previous_period = period;
            }
            previous = mark;
            if (mark >= span.x + .1f*sr) marks[used++] = mark;
        }
        finish_group(start, &used, &count, groups, marks);
    }
    state[0] = i2(count, used);
}

DEVICE float select_value(GLOBAL float *values, int n, int selected) {
    int low = 0, high = n-1;
    while (low < high) {
        float pivot = values[(low+high)/2];
        int left = low, right = high;
        while (left <= right) {
            while (values[left] < pivot) ++left;
            while (values[right] > pivot) --right;
            if (left <= right) {
                float swap = values[left]; values[left] = values[right]; values[right] = swap;
                ++left; --right;
            }
        }
        if (selected <= right) high = right;
        else if (selected >= left) low = left;
        else break;
    }
    return values[selected];
}

KERNEL void group_medians(GLOBAL const float *marks, GLOBAL const int4 *groups,
                            int count, GLOBAL float *scratch, GLOBAL float *medians) {
    int g = get_global_id(0);
    if (g >= count) return;
    int4 group = groups[g]; int n = group.y-1;
    GLOBAL float *values = scratch + group.x;
    for (int i = 0; i < n; ++i) values[i] = marks[group.x+i+1] - marks[group.x+i];
    float median = select_value(values, n, n/2);
    if (!(n & 1)) median = .5f*(median + select_value(values, n, n/2-1));
    medians[g] = median;
}

DEVICE float spline_particular(GLOBAL const float *y, int n, int index) {
    float sum = 0, weight = .2886751345948129f;
    for (int k = 0; k <= 16; ++k) {
        int a = index - k, b = index + k;
        if (a > 0 && a < n - 1) sum += weight * 3 * (y[a + 1] - y[a - 1]);
        if (k && b > 0 && b < n - 1) sum += weight * 3 * (y[b + 1] - y[b - 1]);
        weight *= SPLINE_R;
    }
    return sum;
}

DEVICE float spline_decay(int n) {
    float value = 1;
    for (int i = 0; i < min(n, 32); ++i) value *= SPLINE_R;
    return n > 32 ? 0 : value;
}

KERNEL void spline_edges(GLOBAL const float *y, int n, int rows, GLOBAL float2 *edges) {
    int row = get_global_id(0);
    if (row >= rows) return;
    GLOBAL const float *v = y + row * n;
    float left = .5f * (5 * (v[1] - v[0]) + v[2] - v[1]);
    float right = .5f * (5 * (v[n-1] - v[n-2]) + v[n-2] - v[n-3]);
    left -= spline_particular(v, n, 0) + 2 * spline_particular(v, n, 1);
    right -= 2 * spline_particular(v, n, n-2) + spline_particular(v, n, n-1);
    float c = 1 + 2 * SPLINE_R, e = spline_decay(n-2) * (2 + SPLINE_R);
    edges[row] = v2(c * left - e * right, c * right - e * left) / (c*c - e*e);
}

KERNEL void spline_slopes(GLOBAL const float *y, int n, int rows,
                            GLOBAL const float2 *edges, GLOBAL float *slopes) {
    int i = get_global_id(0), row = get_global_id(1);
    if (i >= n || row >= rows) return;
    // 一様格子の三重対角逆行列は|r|=0.268で減衰するため、16項でfloat32の丸め誤差以下になる。
    // 両端の同次解を加え、CPUのnot-a-knot境界条件も維持する。
    float2 edge = edges[row];
    slopes[row*n+i] = spline_particular(y + row*n, n, i)
        + edge.x * spline_decay(i) + edge.y * spline_decay(n-1-i);
}

DEVICE float hermite(float y0, float y1, float d0, float d1, float t) {
    float difference = y1 - y0;
    return y0 + t * (d0 + t * (3*difference - 2*d0 - d1 + t * (d0+d1-2*difference)));
}

DEVICE float sample_wave(GLOBAL const float *y, GLOBAL const float *slopes, int n,
                  int base, float relative) {
    int step = (int)floor(relative), index = clamp(base + step, 0, n-2);
    float fraction = (base + step - index) + (relative - step);
    return hermite(y[index], y[index+1], slopes[index], slopes[index+1], fraction);
}

DEVICE float2 complex_product(float2 a, float2 b) {
    return v2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

DEVICE int reverse_bits(int value) {
    int result = 0;
    for (int i = 0; i < 11; ++i) { result = (result << 1) | (value & 1); value >>= 1; }
    return result;
}

DEVICE void fft_fixed(LOCAL_PTR float2 *values, int lane, int inverse) {
    for (int size = 2; size <= FFT_SIZE; size *= 2) {
        for (int pair = lane; pair < FFT_HALF; pair += LANES) {
            int at = pair % (size/2), left = (pair / (size/2)) * size + at;
            float angle = (inverse ? 2.0f : -2.0f) * at / size;
            float2 phase = v2(cospi(angle), sinpi(angle));
            float2 a = values[left], b = complex_product(values[left+size/2], phase);
            values[left] = a+b; values[left+size/2] = a-b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

KERNEL void phase_spectra(GLOBAL const float *wave, GLOBAL const float *slopes,
                            int n, GLOBAL const float *marks, int first, int frames,
                            GLOBAL float *original, GLOBAL float2 *spectra) {
    int row = get_group_id(0), lane = get_local_id(0);
    if (row >= frames) return;
    LOCAL float2 values[FFT_SIZE];
    int center = first + row + 1;
    float mark = marks[center]; int base = (int)floor(mark);
    float left = mark - marks[center-1], right = marks[center+1] - mark;
    for (int i = lane; i < FFT_SIZE; i += LANES) {
        float phase = (float)i / FFT_HALF - 1;
        float point = sample_wave(wave, slopes, n, base, mark - base + phase * (phase < 0 ? left : right));
        float value = point * (.5f + .5f * cospi(phase));
        original[row*FFT_SIZE+i] = value;
        values[reverse_bits(i)] = v2(value, 0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    fft_fixed(values, lane, 0);
    for (int i = lane; i <= FFT_HALF; i += LANES) spectra[row*(FFT_HALF+1)+i] = values[i];
}

DEVICE int reflect_index(int index, int n) {
    int value = index % (2*n);
    if (value < 0) value += 2*n;
    return value >= n ? 2*n-1-value : value;
}

KERNEL void magnitude_trend(GLOBAL const float2 *spectra, int rows, int low, int count,
                              GLOBAL const float *logs, GLOBAL const float *weights,
                              int radius, GLOBAL float2 *adjusted) {
    int bin = get_global_id(0), row = get_global_id(1);
    if (bin > FFT_HALF || row >= count) return;
    int selected = low + row;
    float trend = 0;
    for (int k = -radius; k <= radius; ++k) {
        int source = reflect_index(selected+k, rows);
        trend += weights[k+radius] * logs[source*(FFT_HALF+1)+bin];
    }
    float2 value = spectra[selected*(FFT_HALF+1)+bin];
    float change = trend - logs[selected*(FFT_HALF+1)+bin];
    adjusted[row*(FFT_HALF+1)+bin] = value * exp(.85f * clamp(change, -.4605170186f, .4605170186f));
}

KERNEL void log_magnitudes(GLOBAL const float2 *spectra, int total, GLOBAL float *logs) {
    int i = get_global_id(0);
    if (i >= total) return;
    float2 value = spectra[i];
    logs[i] = log(fmax(hypot(value.x, value.y), 1e-12f));
}

KERNEL void gaussian_weights(float sigma, int radius, GLOBAL float *weights) {
    if (get_global_id(0)) return;
    float total = 0;
    for (int k = -radius; k <= radius; ++k) {
        float value = exp(-.5f*k*k/(sigma*sigma));
        weights[k+radius] = value; total += value;
    }
    for (int k = 0; k <= 2*radius; ++k) weights[k] /= total;
}

KERNEL void phase_delta(GLOBAL const float2 *spectra, GLOBAL const float *original,
                          int low, int frames, GLOBAL float *delta) {
    int row = get_group_id(0), lane = get_local_id(0);
    if (row >= frames) return;
    LOCAL float2 values[FFT_SIZE];
    LOCAL float2 energy[LANES];
    for (int i = lane; i < FFT_SIZE; i += LANES) {
        float2 v = spectra[row*(FFT_HALF+1)+(i <= FFT_HALF ? i : FFT_SIZE-i)];
        if (i > FFT_HALF) v.y = -v.y;
        values[reverse_bits(i)] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    fft_fixed(values, lane, 1);
    float2 sum = v2(0);
    for (int i = lane; i < FFT_SIZE; i += LANES) {
        float a = original[(row+low)*FFT_SIZE+i], b = values[i].x / FFT_SIZE;
        sum += v2(a*a, b*b);
    }
    energy[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int size = LANES/2; size > 0; size /= 2) {
        if (lane < size) energy[lane] += energy[lane+size];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = sqrt(energy[0].x / fmax(energy[0].y, 1e-24f));
    for (int i = lane; i < FFT_SIZE; i += LANES)
        delta[row*FFT_SIZE+i] = values[i].x / FFT_SIZE * scale - original[(row+low)*FFT_SIZE+i];
}

KERNEL void overlap_add(GLOBAL const float *delta, GLOBAL const float *slopes,
                          GLOBAL const float *marks, int first, int frames, int parity,
                          GLOBAL float *numerator, GLOBAL float *denominator) {
    int row = 2 * get_global_id(1) + parity, offset = get_global_id(0);
    if (row >= frames) return;
    int center = first + row + 1;
    int sample = (int)ceil(marks[center-1]) + offset;
    if (sample >= (int)floor(marks[center+1])) return;
    float relative = sample - marks[center];
    float phase = relative / (relative < 0 ? marks[center]-marks[center-1] : marks[center+1]-marks[center]);
    float value = sample_wave(delta + row*FFT_SIZE, slopes + row*FFT_SIZE, FFT_SIZE, 0, (phase+1)*FFT_HALF);
    float window = .5f + .5f*cospi(phase);
    // 同じ偶奇の窓は重ならない。二回に分け、浮動小数のatomic加算順による揺れを避ける。
    numerator[sample] += window * value;
    denominator[sample] += window * window;
}

DEVICE float fade_samples(float offset, float length, float sr) {
    float ramp = clamp(fmin(offset, length-offset) / (.025f * sr), 0.0f, 1.0f);
    return .5f - .5f*cospi(ramp);
}

KERNEL void apply_magnitude(GLOBAL const float *wave, GLOBAL float *result,
                              GLOBAL const float *numerator, GLOBAL const float *denominator,
                              GLOBAL const float *marks, int first, int count, float sr) {
    float left = marks[first+1], right = marks[first+count-2];
    int sample = (int)ceil(left) + get_global_id(0);
    if (sample >= (int)floor(right)) return;
    float changed = denominator[sample] > 1e-12f ? numerator[sample]/denominator[sample] : 0;
    result[sample] = wave[sample] + fade_samples(sample-left, right-left, sr) * changed;
}

KERNEL void log_periods(GLOBAL const float *marks, int first, int count, GLOBAL float *logs) {
    int i = get_global_id(0);
    if (i < count-1) logs[i] = log(marks[first+i+1] - marks[first+i]);
}

KERNEL void target_periods(GLOBAL const float *logs, int count,
                             GLOBAL const float *weights, int radius, GLOBAL float *target) {
    int i = get_global_id(0);
    if (i >= count-1) return;
    float trend = 0;
    for (int k = -radius; k <= radius; ++k)
        trend += weights[k+radius] * logs[reflect_index(i+k, count-1)];
    float period = exp(logs[i]);
    target[i] = clamp(exp(.5f*(logs[i]+trend)), .8f*period, 1.2f*period);
}

KERNEL void period_positions(GLOBAL const float *marks, int first, int count,
                               GLOBAL const float *target, GLOBAL float *destination) {
    if (get_global_id(0)) return;
    float total = 0, remainder = 0;
    for (int i = 0; i < count-1; ++i) {
        float part = target[i]-remainder, next = total+part;
        remainder = (next-total)-part; total = next;
    }
    float length = marks[first+count-1] - marks[first], scale = length/total;
    total = 0; remainder = 0; destination[0] = 0;
    for (int i = 0; i < count-1; ++i) {
        float part = target[i]*scale-remainder, next = total+part;
        remainder = (next-total)-part; total = next; destination[i+1] = total;
    }
    destination[count-1] = length;
}

KERNEL void inverse_slopes(GLOBAL const float *marks, int first, int count,
                             GLOBAL const float *destination, GLOBAL float *slopes) {
    int i = get_global_id(0);
    if (i >= count) return;
    int left = max(0, min(i-1, count-3));
    float h0 = destination[left+1]-destination[left], h1 = destination[left+2]-destination[left+1];
    float d0 = (marks[first+left+1]-marks[first+left])/h0;
    float d1 = (marks[first+left+2]-marks[first+left+1])/h1;
    if (i == 0) slopes[i] = fmax(0.0f, ((2*h0+h1)*d0-h0*d1)/(h0+h1));
    else if (i == count-1) slopes[i] = fmax(0.0f, ((2*h1+h0)*d1-h1*d0)/(h0+h1));
    else {
        float w0 = 2*h1+h0, w1 = h1+2*h0;
        slopes[i] = (w0+w1)/(w0/d0+w1/d1);
    }
}

KERNEL void apply_periods(GLOBAL const float *wave, GLOBAL const float *wave_slopes, int n,
                            GLOBAL const float *marks, int first, int count,
                            GLOBAL const float *destination, GLOBAL const float *inverse,
                            float sr, GLOBAL float *result) {
    int base = (int)floor(marks[first]);
    int sample = (int)ceil(marks[first]) + get_global_id(0);
    if (sample >= (int)floor(marks[first+count-1])) return;
    float position = (sample-base)-(marks[first]-base);
    int low = 0, high = count-1;
    while (high-low > 1) {
        int middle = (low+high)/2;
        if (position < destination[middle]) high = middle;
        else low = middle;
    }
    float h = destination[high]-destination[low], t = (position-destination[low])/h;
    float original = marks[first+low], interval = marks[first+high]-original;
    int source = (int)floor(original);
    float relative = original-source + hermite(0, interval, inverse[low]*h, inverse[high]*h, t);
    float reconstructed = sample_wave(wave, wave_slopes, n, source, relative);
    float mask = fade_samples(position, destination[count-1], sr);
    result[sample] = wave[sample] + mask*(reconstructed-wave[sample]);
}

KERNEL void wave_energy(GLOBAL const float *original, GLOBAL const float *result,
                          int n, GLOBAL float2 *parts) {
    int lane = get_local_id(0), group = get_group_id(0);
    LOCAL float2 energy[LANES];
    float2 sum = v2(0);
    for (int i = group*LANES*4+lane; i < min(n, (group+1)*LANES*4); i += LANES) {
        float a = original[i], b = result[i]; sum += v2(a*a, b*b);
    }
    energy[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int size = LANES/2; size > 0; size /= 2) {
        if (lane < size) energy[lane] += energy[lane+size];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (!lane) parts[group] = energy[0];
}

KERNEL void energy_scale(GLOBAL const float2 *parts, int count, GLOBAL float *scale) {
    if (get_global_id(0)) return;
    float2 total = v2(0);
    for (int i = 0; i < count; ++i) total += parts[i];
    scale[0] = sqrt(total.x/total.y);
}

KERNEL void normalize_wave(GLOBAL float *wave, int n, GLOBAL const float *scale) {
    int i = get_global_id(0);
    if (i < n) wave[i] *= scale[0];
}
