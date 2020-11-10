#include <stdio.h>
#include <stdlib.h>

#include "xTATAx.c"
#include "xTATAx_fwd.c"

#include <time.h>

#define BILLION 1000000000;

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        return 0;
    }
    int num_iter = 10;
    int N = atoi(argv[1]);

    float *x = malloc(sizeof(float) * N);
    float *A = malloc(sizeof(float) * N * N);
    float *Ad = malloc(sizeof(float) * N * N);
    float out = 1, outd = 1;
    for (int i = 0; i < N; i++) {
        x[i] = (float)(i + 1) / N;
    }
    for (int i = 0; i < N * N; i++) {
        A[i] = Ad[i] = (float)(i + 1) / N;
    }

    long double fwd_time = 1e20;
    long double deriv_time = 1e20;
    int i = 0;
    for (i = 0; i < num_iter; i++) {
        struct timespec start, fwd, end;
        clock_gettime(CLOCK_REALTIME, &start);
        xTATAx(x, A, &out, N);
        clock_gettime(CLOCK_REALTIME, &fwd);
        // Computing forward derivative of some matrix Ad
        xTATAx_d(x, A, Ad, &out, &outd, N);
        clock_gettime(CLOCK_REALTIME, &end);
        long double this_fwd_time = (long double)(fwd.tv_sec - start.tv_sec)
            + (long double)(fwd.tv_nsec - start.tv_nsec) / (long double)BILLION;
        long double this_deriv_time = (long double)(end.tv_sec - fwd.tv_sec)
            + (long double)(end.tv_nsec - fwd.tv_nsec) / (long double)BILLION;
        if (this_fwd_time < fwd_time) {
            fwd_time = this_fwd_time;
        }
        if (this_deriv_time < deriv_time) {
            deriv_time = this_deriv_time;
        }
    }
    printf("Minimum forward time:%Lf\n", fwd_time);
    printf("Minimum derivative time:%Lf\n", deriv_time);
    printf("Ratio:%Lf\n", deriv_time / fwd_time);

    free(Ad);
    free(A);
    free(x);
}
