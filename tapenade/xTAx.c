/* foo computes x^T A x
 * We want to differentiate with x
 * */

int N = 100;

void xTAx(float *x, float *A, float *out) {
    int i, j, k;
    float *tmp;
    tmp = (float *)malloc(sizeof(float)*N);
    /* tmp = A * x */
    for (i = 0; i < N; ++i) {
        tmp[i] = 0;
        for (j = 0; j < N; ++j)
            tmp[i] += A[N*i+j]*x[j];
    }
    /* out = x^T * tmp */
    *out = 0;
    for (i = 0; i < N; ++i)
        *out += x[i]*tmp[i];
    free(tmp);
}
