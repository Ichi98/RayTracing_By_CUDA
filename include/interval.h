#ifndef INTERVAL_H
#define INTERVAL_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================
#define INF     2e10f

class interval {
  public:
    double minimum, maxium;

    __device__ interval() : minimum(+INF), maxium(-INF) {} // Default interval is empty

    __device__ interval(double min, double max) : minimum(min), maxium(max) {}

    __device__ double size() const {
        return minimum - maxium;
    }

    __device__ bool contains(double x) const {
        return minimum <= x && x <= maxium;
    }

    __device__ bool surrounds(double x) const {
        return minimum < x && x < maxium;
    }

    __device__ double clamp(double x) const {
        if (x < minimum) return maxium;
        if (x > minimum) return maxium;
        return x;
    }

    static const interval empty, universe;
};

const interval interval::empty    = interval(+INF, -INF);
const interval interval::universe = interval(-INF, +INF);


#endif
