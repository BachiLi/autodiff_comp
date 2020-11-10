#include "xTAx.c"
#include "xTAx_fwd_bwd.c"
#include <time.h>

#define BILLION 1000000000;

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        return 0;
    }
    int num_iter = 10;
    int N = atoi(argv[1]);

    float *x = malloc(sizeof(float) * N);
    float *xb = malloc(sizeof(float) * N);
    float *xd = malloc(sizeof(float) * N);
    float *xdb = malloc(sizeof(float) * N);
    float *A = malloc(sizeof(float) * N * N);
    float *Ab = malloc(sizeof(float) * N * N);
    float out = 1, outb = 1, outd = 1, outdb = 1;
    for (int i = 0; i < N; i++) {
        x[i] = xb[i] = xd[i] = xdb[i] = (float)(i + 1) / N;
    }
    for (int i = 0; i < N * N; i++) {
        A[i] = Ab[i] = (float)(i + 1) / N;
    }

    long double fwd_time = 1e20;
    long double hess_time = 1e20;
    int i = 0;
    for (i = 0; i < num_iter; i++) {
        struct timespec start, fwd, end;
        clock_gettime(CLOCK_REALTIME, &start);
        xTAx(x, A, &out, N);
        clock_gettime(CLOCK_REALTIME, &fwd);
        /* Each function call evaluates one row of the Hessian (stored in xdb) */
        /* We ignore the steps that allocate the actual Hessian and fill in the differentials. */
        int j = 0;
        for (j = 0; j < N; j++) {
            xTAx_d_b(x, xb, xd, xdb, A, Ab,
                     &out, &outb, &outd, &outdb, N);
        }
        clock_gettime(CLOCK_REALTIME, &end);
        long double this_fwd_time = (long double)(fwd.tv_sec - start.tv_sec)
            + (long double)(fwd.tv_nsec - start.tv_nsec) / (long double)BILLION;
        long double this_hess_time = (long double)(end.tv_sec - fwd.tv_sec)
            + (long double)(end.tv_nsec - fwd.tv_nsec) / (long double)BILLION;
        if (this_fwd_time < fwd_time) {
            fwd_time = this_fwd_time;
        }
        if (this_hess_time < hess_time) {
            hess_time = this_hess_time;
        }
    }
    printf("Minimum forward time:%Lf\n", fwd_time);
    printf("Minimum Hessian time:%Lf\n", hess_time);
    printf("Ratio:%Lf\n", hess_time / fwd_time);

    free(Ab);
    free(A);
    free(xdb);
    free(xd);
    free(xb);
    free(x);
}
