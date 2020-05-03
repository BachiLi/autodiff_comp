#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <time.h>
#include <stdio.h>

typedef unsigned char bool;

void arap(int W, int H, float w_fit, float w_reg,
          const float *Offsets_Angle, /* [W, H, 3] */
          const float *UrShape, /* [W, H, 2] */
          const float *Constraints, /* [W, H, 2] */
          const bool *Mask, /* [W, H] */
          const bool *C_valid, /* [W, H] */
          int i, int j,
          float *Output /* [10] */);

void arap_b(int W, int H, float w_fit, float w_reg, const float *Offsets_Angle
        , float *Offsets_Angleb, const float *UrShape, const float *
        Constraints, const bool *Mask, const bool *C_valid, int i, int j, 
        float *Output, float *Outputb);

void arap_d(int W, int H, float w_fit, float w_reg, const float *Offsets_Angle
        , const float *Offsets_Angled, const float *UrShape, const float *
        Constraints, const bool *Mask, const bool *C_valid, int i, int j, 
        float *Output, float *Outputd);

typedef struct Vec3f {
    float x, y, z;
} Vec3f;

float dot(Vec3f a, Vec3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3f get_vec3f(int W, int H,
                const float *img,
                int i, int j) {
    if (i < 0 || i >= W || j < 0 || j >= H) {
        Vec3f zero;
        zero.x = 0;
        zero.y = 0;
        zero.z = 0;
        return zero;
    }
    Vec3f ret;
    ret.x = img[3 * (i * H + j) + 0];
    ret.y = img[3 * (i * H + j) + 1];
    ret.z = img[3 * (i * H + j) + 2];
    return ret;
}

int main(int argc, char *argv[]) {
    int width, height, channels_in_file;
    unsigned char *transposed_Mask = stbi_load("../data/mask0.png", &width, &height, &channels_in_file, 1);
    unsigned char *Mask = malloc(sizeof(unsigned char) * width * height * channels_in_file);
    // Transpose the loaded mask
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            Mask[x * height + y] = transposed_Mask[y * width + x];
        }
    }
    free(transposed_Mask);
    // Initialize the inputs & outputs
    float *Offsets_Angle = malloc(sizeof(float) * width * height * 3);
    float *UrShape = malloc(sizeof(float) * width * height * 2);
    float *Constraints = malloc(sizeof(float) * width * height * 2);
    unsigned char *C_valid = malloc(sizeof(unsigned char) * width * height);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            Offsets_Angle[3 * (x * height + y) + 0] = 0.5f;
            Offsets_Angle[3 * (x * height + y) + 1] = 0.5f;
            Offsets_Angle[3 * (x * height + y) + 2] = 0.5f;
            UrShape[2 * (x * height + y) + 0] = 0.5f;
            UrShape[2 * (x * height + y) + 1] = 0.5f;
            Constraints[2 * (x * height + y) + 0] = 0.5f;
            Constraints[2 * (x * height + y) + 1] = 0.5f;
            C_valid[(x * height + y)] = 1;
        }
    }
    float *Output = malloc(sizeof(float) * width * height * 10);

    int num_trials = 10;
    clock_t start_t, end_t;
    // Forward only
    start_t = clock();
    for (int i = 0; i < num_trials; i++) {
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                arap(width, height, 0.5f, 0.5f,
                     Offsets_Angle, UrShape, Constraints, Mask, C_valid, x, y,
                     &Output[10 * (x * height + y)]);
            }
        }
    }
    end_t = clock();
    printf("Time spent on forward pass: %lf\n", (double)(end_t - start_t) / (num_trials * CLOCKS_PER_SEC));

    // J^T * J * val
    float *val = malloc(sizeof(float) * width * height * 3);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            val[3 * (x * height + y)] = 0.5f;
            val[3 * (x * height + y) + 1] = 0.5f;
            val[3 * (x * height + y) + 2] = 0.5f;
        }
    }
    float *next_val = malloc(sizeof(float) * width * height * 10);
    // Buffer for tangent/adjoint computation
    float *tangent = malloc(sizeof(float) * width * height * 3);
    memset(tangent, 0, sizeof(float) * width * height * 3);
    float *adjoint = malloc(sizeof(float) * width * height * 3);
    memset(adjoint, 0, sizeof(float) * width * height * 3);
    float Output_dummy[10];
    float Outputd[10];
    memset(Outputd, 0, sizeof(float) * 10);
    float Outputb[10];
    memset(Outputb, 0, sizeof(float) * 10);
    start_t = clock();
    for (int i = 0; i < num_trials; i++) {
        // We first compute next_val = J*val
        // J has [W, H, 3] columns and [W, H, 10] rows
        // Each call to arap_b computes one row of the Jacobian
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                for (int j = 0; j < 10; j++) {
                    Outputb[j] = 1;
                    arap_b(width, height, 0.5f, 0.5f,
                           Offsets_Angle, adjoint, UrShape,
                           Constraints, Mask, C_valid, x, y,
                           Output_dummy,
                           Outputb);
                    Outputb[j] = 0;

                    // The Jacobian is sparse -- only need to check (x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)
                    // Do dot product with val
                    next_val[10 * (x * height + y) + j] =
                        dot(get_vec3f(width, height, adjoint, x, y),
                            get_vec3f(width, height, val, x, y)) +
                        dot(get_vec3f(width, height, adjoint, x + 1, y),
                            get_vec3f(width, height, val, x + 1, y)) +
                        dot(get_vec3f(width, height, adjoint, x - 1, y),
                            get_vec3f(width, height, val, x - 1, y)) +
                        dot(get_vec3f(width, height, adjoint, x, y + 1),
                            get_vec3f(width, height, val, x, y + 1)) +
                        dot(get_vec3f(width, height, adjoint, x, y - 1),
                            get_vec3f(width, height, val, x, y - 1));

                }
            }
        }
        // Now we compute val = J^T * next_val
        // Each call to arap_d computes one column of the Jacobian
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                for (int j = 0; j < 3; j++) {
                    float sum = 0.f;
                    tangent[3 * (x * height + y) + j] = 1;
                    for (int l = 0; l < 5; l++) {
                        int dx[] = {0, -1, 1,  0, 0};
                        int dy[] = {0,  0, 0, -1, 1};
                        arap_d(width, height, 0.5f, 0.5f,
                               Offsets_Angle, tangent, UrShape,
                               Constraints, Mask, C_valid, x + dx[l], y + dy[l],
                               Output_dummy,
                               Outputd);
                        // dot(Outputd, next_val)
                        for (int k = 0; k < 10; k++) {
                            sum += Outputd[k] * next_val[10 * ((x + dx[l]) * height + (y + dy[l])) + k];
                        }
                    }
                    val[3 * (x * height + y) + j] = sum;
                    tangent[3 * (x * height + y) + j] = 0;
                }
            }
        }
    }
    end_t = clock();
    printf("Time spent on J^TJx: %lf\n", (double)(end_t - start_t) / (num_trials * CLOCKS_PER_SEC));

    // J^T * Output
    for (int i = 0; i < num_trials; i++) {
        // First compute Output
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                arap(width, height, 0.5f, 0.5f,
                     Offsets_Angle, UrShape, Constraints, Mask, C_valid, x, y,
                     &Output[10 * (x * height + y)]);
            }
        }
        // Next we compute val = J^T * Output
        // Each call to arap_d computes one column of the Jacobian
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                for (int j = 0; j < 3; j++) {
                    float sum = 0.f;
                    tangent[3 * (x * height + y) + j] = 1;
                    for (int l = 0; l < 5; l++) {
                        int dx[] = {0, -1, 1,  0, 0};
                        int dy[] = {0,  0, 0, -1, 1};
                        arap_d(width, height, 0.5f, 0.5f,
                               Offsets_Angle, tangent, UrShape,
                               Constraints, Mask, C_valid, x + dx[l], y + dy[l],
                               Output_dummy,
                               Outputd);
                        // dot(Outputd, next_val)
                        for (int k = 0; k < 10; k++) {
                            sum += Outputd[k] * Output[10 * ((x + dx[l]) * height + (y + dy[l])) + k];
                        }
                    }
                    val[3 * (x * height + y) + j] = sum;
                    tangent[3 * (x * height + y) + j] = 0;
                }
            }
        }
    }
    end_t = clock();
    printf("Time spent on J^T * F(x): %lf\n", (double)(end_t - start_t) / (num_trials * CLOCKS_PER_SEC));

    free(val);
    free(next_val);
    free(tangent);
    free(adjoint);
    free(Output);
    free(Offsets_Angle);
    free(UrShape);
    free(Constraints);
    free(C_valid);
    free(Mask);

    return 0;
}
