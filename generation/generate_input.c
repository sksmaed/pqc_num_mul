#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <inttypes.h>

#define LIMB_LEN 384
#define DATA_LEN 384
#define P 12289

uint64_t rand64_modp(void) {
    return ((uint64_t)rand() << 32 | rand()) % 10000;
}

int main() {
    srand((unsigned int)time(NULL));

    printf("#ifndef TEST_DATA_H\n#define TEST_DATA_H\n\n#include <stdint.h>\n\n");

    printf("static const uint16_t test_A[%d] = {\n    ", LIMB_LEN);
    for (int i = 0; i < LIMB_LEN; ++i) {
        if (i < DATA_LEN)
            printf("%" PRIu64, rand64_modp() + 1);
        else
            printf("0ULL");

        if (i != LIMB_LEN - 1) printf(", ");
        if ((i + 1) % 4 == 0) printf("\n    ");
    }
    printf("\n};\n\n");

    printf("static const uint64_t test_B[%d] = {\n    ", LIMB_LEN);
    for (int i = 0; i < LIMB_LEN; ++i) {
        if (i < DATA_LEN)
            printf("%" PRIu64, rand64_modp() + 1);
        else
            printf("0ULL");

        if (i != LIMB_LEN - 1) printf(", ");
        if ((i + 1) % 4 == 0) printf("\n    ");
    }
    printf("\n};\n\n");

    printf("#endif\n");
    return 0;
}
