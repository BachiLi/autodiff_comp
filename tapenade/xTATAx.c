/* This computes x^T A^TA x
 * We want to differentiate with A
 * This gives us 2x^T(A^TdA)x
 * Tapanade generates the equivalent of x^T(A^TdA+dA^TA)x
 * */

void xTATx(float *x, float *A, float *out, int N) {
    int i, j, k;
    float *ATA;
    ATA = (float *)malloc(sizeof(float)*N*N);
    float *tmp;
    tmp = (float *)malloc(sizeof(float)*N);
    /* ATA = A^T * A */
    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j) {
            ATA[i*N + j] = 0;
            for (k = 0; k < N; ++k)
                /* A^T[i * N + k] -> A[k * N + i] */
                ATA[i*N + j] += A[k*N+i]*A[k*N+j];
        }
    /* tmp = ATA * x */
    for (i = 0; i < N; ++i) {
        tmp[i] = 0;
        for (j = 0; j < N; ++j)
            tmp[i] += ATA[N*i+j]*x[j];
    }
    /* out = x^T * tmp */
    *out = 0;
    for (i = 0; i < N; ++i)
        *out += x[i]*tmp[i];
    free(tmp);
    free(ATA);
}