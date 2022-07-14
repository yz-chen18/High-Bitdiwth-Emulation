#include <cstdlib>
#include <ctime>

#include "random_sample.h"


// range [6.1*10^-5, 6.55*10^4]
float rand_sample_normal() {
    Fp32 sample{0.f};
    srand(clock());

    // todo: 存在当R进位时，由于小数位全为1，E进位导致E全为1，变为nan，故暂时将E的范围设为 -14~14
    unsigned int E = rand() % 29 + 113;
    unsigned int sign = rand() % 2;
    unsigned int significand = rand() % 0x00800000;

    sample.ui |= (sign << 31);
    sample.ui |= (E << 23);
    sample.ui |= (significand);

    return sample.fp;
}

float rand_sample_positive() {
    Fp32 sample{0.f};
    srand(clock());

    // todo: 存在当R进位时，由于小数位全为1，E进位导致E全为1，变为nan，故暂时将E的范围设为 -14~14
    unsigned int E = rand() % 29 + 113;
    unsigned int significand = rand() % 0x00800000;

    sample.ui |= (0 << 31);
    sample.ui |= (E << 23);
    sample.ui |= (significand);

    return sample.fp;
}

// range [6.1*10^-5, 1]
float rand_sample_zero_to_one() {
    srand(clock());

    return rand() / static_cast<float>(RAND_MAX);
}

float rand_sample_one_to_one() {
    srand(clock());

    int sign = (rand() % 2) * 2 - 1;

    return rand() / static_cast<float>(RAND_MAX) * sign;
}

float rand_sample_one_to_two() {
    srand(clock());

    return rand() / static_cast<float>(RAND_MAX) + 1;
}

float rand_sample(RANGE range) {
    float fp;
    switch (range)
    {
        case NORMAL:
            fp = rand_sample_normal();
            break;
        case POSITIVE:
            fp = rand_sample_positive();
            break;
        case ZERO_TO_ONE:
            fp = rand_sample_zero_to_one();
            break;
        case ONE_TO_ONE:
            fp = rand_sample_one_to_one();
            break;
        case ONE_TO_TWO:
            fp = rand_sample_one_to_two();
            break;
        default:
            break;
    }

    return fp;
}