###############################################################################
#
# Buffer a signal and return samples using time ranges
#
# History:
# 01-03-24 - Levi Burner - Created file
#
###############################################################################

from bisect import bisect, bisect_left
import numpy as np

class SampleBuffer:
    def __init__(self, sample_close=None, sample_interp=None):
        self.t0 = None
        self.ts = []
        self.samples = []
        self.sample_close = sample_close
        self.sample_interp = sample_interp

    def __len__(self):
        return len(self.ts)

    def append(self, t, sample):
        if self.t0:
            t = t - self.t0

        if len(self.ts) > 0:
            assert self.ts[-1] < t

        self.ts.append(t)
        self.samples.append(sample)

    def get(self, t0, tf=None):
        if tf is not None:
            assert tf > t0

            left_idx  = bisect_left(self.ts, t0)
            right_idx = bisect(self.ts, tf, lo=left_idx)

            return self.ts[left_idx:right_idx], self.samples[left_idx:right_idx]
        else:
            left_idx = bisect_left(self.ts, t0)

            if left_idx == len(self.ts):
                return self.ts[-1], self.samples[-1]
            elif left_idx == 0:
                return self.ts[0], self.samples[0]
            elif self.sample_interp is not None:
                return t0, self.sample_interp(t0, self[left_idx-1:left_idx+1])
            else:
                return self.ts[left_idx-1], self.samples[left_idx-1]

    def __getitem__(self, key):
        return self.ts[key], self.samples[key]

    def __setitem__(self, key, value):
        self.ts[key] = value[0]
        self.samples[key] = value[1]

    def __delitem__(self, key):
        del self.ts[key]
        del self.samples[key]

    def trim(self, t0):
        left_idx  = bisect_left(self.ts, t0)

        if self.sample_close:
            for i in range(left_idx):
                self.sample_close(self.ts[i], self.samples[i])

        self.ts = self.ts[left_idx:]
        self.samples = self.samples[left_idx:]

    def set_t0(self, t0=None):
        if t0 is None and self.t0 is None:
            t0_inc = self.ts[-1]
            self.t0 = t0_inc
        elif t0 is None and self.t0 is not None:
            t0_inc = self.ts[-1]
            self.t0 += t0_inc
        elif t0 is not None and self.t0 is None:
            t0_inc = t0
            self.t0 = t0
        elif t0 is not None and self.t0 is not None:
            t0_inc = t0 - self.t0
            self.t0 = t0

        self.ts = (np.array(self.ts) - t0_inc).tolist()

    def set_tf(self, tf):
        if len(self.ts) > 0:
            tf_inc = tf - self.ts[-1]
            self.ts = (np.array(self.ts) + tf_inc).tolist()
        else:
            tf_inc = tf
        if self.t0 is None:
            self.t0 = -tf_inc
        else:
            self.t0 -= tf_inc
        return tf_inc

    def inc_t(self, t_inc):
        if len(self.ts) > 0:
            self.ts = (np.array(self.ts) + t_inc).tolist()

    def samples_since(self, t):
        left_idx = bisect_left(self.ts, t)
        return (len(self.ts) - 1) - left_idx

