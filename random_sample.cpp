#include <cstdlib>
#include <ctime>

#include "common.h"

float rand_sample() {
    Fp32 sample{0.f};
    srand(clock());

    unsigned int E = rand() % 30 + 112;
    unsigned int sign = rand() % 1;
    unsigned int significand = rand() % 0x007fffff;

    sample.ui |= (sign << 31);
    sample.ui |= (E << 23);
    sample.ui |= (significand);

    return sample.fp;
}