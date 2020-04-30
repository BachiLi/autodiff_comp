#include <math.h>

struct Vec2f {
    float x, y;
};

Vec2f sq(Vec2f v) {
    Vec2f ret;
    ret.x = v.x * v.x;
    ret.y = v.y * v.y;
    return ret;
}

Vec2f sub(Vec2f a, Vec2f b) {
    Vec2f ret;
    ret.x = a.x - b.x;
    ret.y = a.y - b.y;
    return ret;
}

Vec2f get_vec2f_unchecked(int W, int H, int C,
                          const float *img,
                          int i, int j) {
    Vec2f ret;
    ret.x = Offsets_Angle[C * (i * H + j) + 0];
    ret.y = Offsets_Angle[C * (i * H + j) + 1];
    return ret;
}

Vec2f get_vec2f(int W, int H, int C,
                const float *img,
                int i, int j) {
    if (i < 0 || i >= W || j < 0 || j >= H) {
        return 0.f;
    }
    Vec2f ret;
    ret.x = Offsets_Angle[C * (i * H + j) + 0];
    ret.y = Offsets_Angle[C * (i * H + j) + 1];
    return ret;
}

bool get_mask_unchecked(int W, int H,
                        const bool *img,
                        int i, int j) {
    return img[i * H + j];
}

bool get_mask(int W, int H,
              const bool *img,
              int i, int j) {
    if (i < 0 || i >= W || j < 0 || j >= H) {
        return false;
    }
    return img[i * H + j];
}

Vec2f Rot2D(float a, Vec2f v) {
    Vec2f ret;
    ret.x = cosf(a) * v.x - sinf(a) * v.y;
    ret.y = sinf(a) * v.x + cosf(a) * v.y;
    return ret;
}

Vec2f reg(int W, int H,
          const float *Offsets_Angle,
          const float *UrShape,
          const bool *Mask,
          int i, int j,
          int x, int y) {
    if (get_mask(W, H, Mask, i, j) && get_mask(W, H, Mask, i + x, j + y)) {
        Vec2f d_off = sub(get_vec2f_unchecked(W, H, 3, Offsets_Angle, i, j),
                          get_vec2f(W, H, Offsets_Angle, i + x, i + y));
        Vec2f d_ur = sub(get_vec2f_unchecked(W, H, 2, UrShape, i, j),
                         get_vec2f(W, H, 2, UrShape, i + x, j + y));
        Vec2f d_diff = sub(d_off, Rot2D(Offsets_Angle[3 * (i * H + j) + 2], d_ur));
        return sq(d_diff);
    } else {
        Vec2f zero;
        zero.x = 0;
        zero.y = 0;
        return zero;
    }
}

void arap(int W, int H, float w_fit, float w_reg,
          const float *Offsets_Angle, /* [W, H, 3] */
          const float *UrShape, /* [W, H, 2] */
          const float *Constraints, /* [W, H, 2] */
          const bool *Mask, /* [W, H] */
          const bool *C_valid, /* [W, H] */
          float *Output /* [W, H, 10] */) {
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            if (C_valid[i * W + j]) {
                Vec2f E_fit = sq(sub(get_vec2f_unchecked(W, H, 3, Offsets_Angle, i, j),
                                     get_vec2f_unchecked(W, H, 2, Constraints, i, j)));
                Output[10 * (i * H + j) + 0] = w_fit * E_fit.x;
                Output[10 * (i * H + j) + 1] = w_fit * E_fit.y;
            } else {
                Output[10 * (i * H + j) + 0] = 0;
                Output[10 * (i * H + j) + 1] = 0;
            }
            Vec2f reg_right = reg(W, H, Offsets_Angle, UrShape, Mask, i, j, 1, 0);
            Vec2f reg_left = reg(W, H, Offsets_Angle, UrShape, Mask, i, j, -1, 0);
            Vec2f reg_bottom = reg(W, H, Offsets_Angle, UrShape, Mask, i, j, 0, 1);
            Vec2f reg_top = reg(W, H, Offsets_Angle, UrShape, Mask, i, j, 0, -1);
            Output[10 * (i * H + j) + 2] = w_reg * reg_right.x;
            Output[10 * (i * H + j) + 3] = w_reg * reg_right.y;
            Output[10 * (i * H + j) + 4] = w_reg * reg_left.x;
            Output[10 * (i * H + j) + 5] = w_reg * reg_left.y;
            Output[10 * (i * H + j) + 6] = w_reg * reg_bottom.x;
            Output[10 * (i * H + j) + 7] = w_reg * reg_bottom.y;
            Output[10 * (i * H + j) + 8] = w_reg * reg_top.x;
            Output[10 * (i * H + j) + 9] = w_reg * reg_top.y;
        }
    }
}
