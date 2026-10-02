// 正規化相互相関と経路探索による周期検出。音声長・サンプリング周波数は実行時の引数とする。
#define CANDIDATES 15

DEVICE float sinc_corr(GLOBAL const float *corr, int window, float x, int depth) {
    int left = (int)floor(x);
    float fraction = x - left;
    if (fraction == 0) return corr[abs(left)];
    depth = min(depth, window - left);
    if (depth <= 1) return mix(corr[abs(left)], corr[abs(left + 1)], fraction);
    float sum = 0;
    for (int k = 1 - depth; k <= depth; ++k) {
        float distance = fraction - k;
        float radius = k <= 0 ? depth + fraction : depth + 1 - fraction;
        float weight = sinpi(distance) / (M_PI_F * distance)
            * .5f * (1 + cospi(distance / radius));
        sum += corr[abs(left + k)] * weight;
    }
    return sum;
}

KERNEL void global_peak(GLOBAL const float *wave, int n, GLOBAL float *peak) {
    int lane = get_local_id(0);
    LOCAL float sums[256];
    LOCAL float2 maxima[256];
    float sum = 0;
    for (int i = lane; i < n; i += 256) sum += wave[i];
    sums[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int size = 128; size > 0; size /= 2) {
        if (lane < size) sums[lane] += sums[lane+size];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = sums[0]/n;
    float maximum = 0, absolute = 0;
    for (int i = lane; i < n; i += 256) {
        maximum = fmax(maximum, fabs(wave[i] - mean));
        absolute = fmax(absolute, fabs(wave[i]));
    }
    maxima[lane] = v2(maximum, absolute);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int size = 128; size > 0; size /= 2) {
        if (lane < size) maxima[lane] = fmax(maxima[lane], maxima[lane+size]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (!lane) { peak[0] = maxima[0].x; peak[1] = maxima[0].y; }
}

KERNEL void frame_statistics(GLOBAL const float *wave, int n, float sr, float first,
                               int nf, int period, int window, GLOBAL const float *peak,
                               GLOBAL float2 *statistics, GLOBAL float *intensity) {
    int frame = get_global_id(0);
    if (frame >= nf) return;
    float center = first + frame * 0.005f * sr;
    int middle = (int)floor(center);
    float mean = 0;
    for (int i = middle + 1 - period; i <= middle + period; ++i)
        mean += wave[clamp(i, 0, n - 1)];
    mean /= 2 * period;
    float p = 0, xx = 0;
    for (int i = middle + 1 - window / 2; i <= middle + window / 2; ++i)
        p = fmax(p, fabs(wave[clamp(i, 0, n - 1)] - mean));
    intensity[frame] = peak[0] > 0 ? fmin(p / peak[0], 1.0f) : 0;
    int start = max(0, (int)floor(center - sr/65.0f));
    for (int j = 0; j < window; ++j) {
        float x = wave[start+j]-mean; xx += x*x;
    }
    statistics[frame] = v2(mean, xx);
}

KERNEL void correlations(GLOBAL const float *wave, int n, float sr,
                           float first, int nf, int window, GLOBAL const float2 *statistics,
                           GLOBAL float *corr) {
    int lag = get_global_id(0), frame = get_global_id(1);
    if (lag > window || frame >= nf) return;
    float center = first + frame * 0.005f * sr;
    float mean = statistics[frame].x, xx = statistics[frame].y;
    int start = max(0, (int)floor(center - sr / 65.0f));
    float xy = 0, yy = 0;
    for (int j = 0; j < window; ++j) {
        float x = wave[start + j] - mean;
        float y = wave[min(start + j + lag, n - 1)] - mean;
        xy += x * y; yy += y * y;
    }
    corr[frame * (window + 1) + lag] = xx * yy > 0 ? xy / sqrt(xx * yy) : 0;
}

KERNEL void candidates(GLOBAL const float *corr, GLOBAL const float *intensity,
                         int nf, int window, float sr, GLOBAL float *freq,
                         GLOBAL float *score) {
    int frame = get_global_id(0);
    if (frame >= nf) return;
    int base = frame * CANDIDATES;
    freq[base] = 0;
    score[base] = 0.45f + fmax(0.0f, 2.0f - intensity[frame] * (1.45f / 0.03f));
    for (int i = 1; i < CANDIDATES; ++i) {
        freq[base + i] = 0; score[base + i] = -1e20f;
    }
    for (int lag = 2; lag < window; ++lag) {
        float l = corr[frame * (window + 1) + lag - 1], m = corr[frame * (window + 1) + lag];
        float r = corr[frame * (window + 1) + lag + 1];
        if (m <= .225f || m <= l || m < r) continue;
        float bend = 2 * m - l - r, slope = .5f * (r - l);
        float delta = bend > 0 ? slope / bend : 0;
        float f = sr / (lag + delta);
        if (f <= 0) continue;
        float strength = sinc_corr(corr + frame * (window + 1), window, lag + delta, 30);
        if (strength > 1) strength = 1 / strength;
        float value = strength - .01f * log2(550.0f / f);
        int place = 1;
        for (int i = 2; i < CANDIDATES; ++i)
            if (score[base + i] < score[base + place]) place = i;
        if (value > score[base + place]) {
            freq[base + place] = f; score[base + place] = value;
        }
    }
    // 周波数候補のピークを補間し、サブサンプル精度の周期位置を求める。
    for (int i = 1; i < CANDIDATES; ++i) {
        if (freq[base + i] <= 0) continue;
        float center = rint(sr / freq[base + i]);
        float low = center - 1, high = center + 1;
        for (int step = 0; step < 18; ++step) {
            float a = high - (high - low) * .618034f;
            float b = low + (high - low) * .618034f;
            if (sinc_corr(corr + frame * (window + 1), window, a, 70)
                > sinc_corr(corr + frame * (window + 1), window, b, 70)) high = b;
            else low = a;
        }
        float x = .5f * (low + high), f = sr / x;
        float strength = sinc_corr(corr + frame * (window + 1), window, x, 70);
        if (strength > 1) strength = 1 / strength;
        freq[base + i] = f < 550 ? f : 0;
        score[base + i] = f < 550 ? strength - .01f * log2(550.0f / f) : score[base];
    }
}

// 経路探索はGPU内で完結させ、フレームごとのCPU同期を挟まない。
KERNEL void pitch_path(GLOBAL const float *freq, GLOBAL float *score,
                         int nf, GLOBAL int *trace, GLOBAL float *pitch) {
    if (get_global_id(0)) return;
    for (int t = 1; t < nf; ++t) {
        for (int b = 0; b < CANDIDATES; ++b) {
            float best = -1e30f; int prior = 0;
            float fb = freq[t * CANDIDATES + b];
            for (int a = 0; a < CANDIDATES; ++a) {
                float fa = freq[(t - 1) * CANDIDATES + a];
                float cost = (fa > 0 && fb > 0) ? .7f * fabs(log2(fa / fb))
                    : ((fa > 0) != (fb > 0) ? .28f : 0);
                float value = score[(t - 1) * CANDIDATES + a] - cost;
                if (value > best) { best = value; prior = a; }
            }
            score[t * CANDIDATES + b] += best;
            trace[t * CANDIDATES + b] = prior;
        }
        float maximum = score[t * CANDIDATES];
        for (int i = 1; i < CANDIDATES; ++i) maximum = fmax(maximum, score[t * CANDIDATES + i]);
        for (int i = 0; i < CANDIDATES; ++i) score[t * CANDIDATES + i] -= maximum;
    }
    int choice = 0;
    for (int i = 1; i < CANDIDATES; ++i)
        if (score[(nf - 1) * CANDIDATES + i] > score[(nf - 1) * CANDIDATES + choice]) choice = i;
    for (int t = nf - 1; t >= 0; --t) {
        pitch[t] = freq[t * CANDIDATES + choice];
        choice = trace[t * CANDIDATES + choice];
    }
}

DEVICE float pitch_at(GLOBAL const float *pitch, int nf, float first, float step, float sample) {
    float index = (sample - first) / step;
    int near = (int)floor(index + .5f), left = (int)floor(index);
    if (near < 0 || near >= nf || pitch[near] <= 0) return 0;
    if (left < 0 || left + 1 >= nf || pitch[left] <= 0 || pitch[left + 1] <= 0) return pitch[near];
    return mix(pitch[left], pitch[left + 1], index - left);
}
