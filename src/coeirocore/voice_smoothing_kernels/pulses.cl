// 一つの周期を決める相互相関だけを256レーンで分担し、周期を追う順序は変えない。
DEVICE float2 pulse_correlation(GLOBAL const float *wave, int n, int left, int right, int shift) {
    float xx = 0, yy = 0, xy = 0, peak = 0;
    for (int j = 0; j <= right-left; ++j) {
        int a = left+j, b = shift+j;
        if (a < 0 || a >= n || b < 0 || b >= n) continue;
        float x = wave[a], y = wave[b];
        xx += x*x; yy += y*y; xy += x*y; peak = fmax(peak, fabs(y));
    }
    return v2(xy != 0 ? xy/sqrt(xx*yy) : 0, peak);
}

DEVICE float3 follow_parallel(GLOBAL const float *wave, int n, float at, float period, int direction,
                       LOCAL_PTR float2 *corr, LOCAL_PTR float4 *choices, LOCAL_PTR int *shifts) {
    int lane = get_local_id(0);
    int left = (int)floor(at-.5f*period+.5f), right = (int)floor(at+.5f*period+.5f);
    int low = (int)floor(at + (direction > 0 ? .8f : -1.25f)*period - .5f*period);
    int high = (int)ceil(at + (direction > 0 ? 1.25f : -.8f)*period - .5f*period);
    float4 best = v4(-1, 0, 0, 0);
    int chosen = 2147483647;
    for (int block = low-1; block < high; block += 256) {
        int index = block+lane;
        corr[lane+1] = index < low || index > high ? v2(0) : pulse_correlation(wave, n, left, right, index);
        if (!lane) corr[0] = block-1 < low ? v2(0) : pulse_correlation(wave, n, left, right, block-1);
        if (lane == 255) corr[257] = block+256 > high ? v2(0) : pulse_correlation(wave, n, left, right, block+256);
        barrier(CLK_LOCAL_MEM_FENCE);
        float before = corr[lane].x, middle = corr[lane+1].x, after = corr[lane+2].x;
        if (index < high && middle > best.x && middle >= before && middle >= after) {
            best = v4(middle, before, after, corr[lane+2].y); chosen = index;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    choices[lane] = best; shifts[lane] = chosen;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int size = 128; size > 0; size /= 2) {
        if (lane < size) {
            int other = lane+size;
            if (choices[other].x > choices[lane].x ||
                (choices[other].x == choices[lane].x && shifts[other] < shifts[lane])) {
                choices[lane] = choices[other]; shifts[lane] = shifts[other];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    best = choices[0]; chosen = shifts[0];
    float position = chosen == 2147483647 ? at : at + (chosen-left);
    float bend = 2*best.x-best.y-best.z, value = best.x;
    if (value >= 0 && bend != 0) {
        float slope = .5f*(best.z-best.y);
        position += slope/bend; value += .5f*slope*slope/bend;
    }
    // 次の追跡で共有バッファを書き換える前に、全レーンが今回の結果を受け取る。
    barrier(CLK_LOCAL_MEM_FENCE);
    return v3(position, value, best.w);
}

KERNEL void pulse_track_parallel(GLOBAL const float *wave, int n, float sr, float first,
                                   GLOBAL const float *pitch, int nf, GLOBAL const float *peak,
                                   GLOBAL float *marks) {
    int lane = get_local_id(0);
    LOCAL float2 correlations[258];
    LOCAL float4 choices[256];
    LOCAL int shifts[256];
    LOCAL float shared_anchor;
    float step = .005f*sr, prior_end = -1e20f;
    for (int frame = 0; frame < nf; ++frame) {
        if (pitch[frame] <= 0) continue;
        int end = frame;
        while (end+1 < nf && pitch[end+1] > 0) ++end;
        float lo = first+(frame-.5f)*step, hi = first+(end+.5f)*step;
        float middle = .5f*(lo+hi), period = sr/pitch_at(pitch, nf, first, step, middle);
        if (!lane) {
            int a = max(0, (int)floor(middle-.5f*period)), b = min(n-1, (int)ceil(middle+.5f*period));
            int selected = a;
            for (int i = a+1; i <= b; ++i) if (fabs(wave[i]) > fabs(wave[selected])) selected = i;
            float anchor = selected;
            if (selected > a && selected < b) {
                float bend = 2*wave[selected]-wave[selected-1]-wave[selected+1];
                if (bend != 0) anchor += .5f*(wave[selected+1]-wave[selected-1])/bend;
            }
            shared_anchor = anchor;
            marks[(int)floor(anchor)] = anchor;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int direction = -1; direction <= 1; direction += 2) {
            float position = shared_anchor;
            for (int iteration = 0; iteration < n; ++iteration) {
                float frequency = pitch_at(pitch, nf, first, step, position);
                if (frequency <= 0) break;
                period = sr/frequency;
                float3 next = follow_parallel(wave, n, position, period, direction, correlations, choices, shifts);
                position = next.y < 0 ? position+direction*period : next.x;
                int outside = position < lo || position > hi;
                int accepted = outside ? next.y > .7f && next.z > .023333f*peak[1]
                    : next.y > .3f && (next.z == 0 || next.z > .01f*peak[1]);
                if (direction < 0 && position-prior_end <= .8f*period) accepted = 0;
                if (accepted && position >= 0 && position < n) {
                    if (!lane) marks[(int)floor(position)] = position;
                                if (direction > 0) prior_end = position;
                }
                if (outside) break;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        frame = end;
    }
}
