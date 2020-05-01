/*        Generated by TAPENADE     (INRIA, Ecuador team)
    Tapenade 3.15 (master) - 15 Apr 2020 11:54
*/
#include <math.h>
typedef unsigned char bool;
typedef struct Vec2f {
    float x, y;
} Vec2f;

/*
  Differentiation of sq in forward (tangent) mode:
   variations   of useful results: sq.x sq.y
   with respect to varying inputs: v.x v.y
*/
Vec2f sq_d(Vec2f v, Vec2f vd, Vec2f *sq) {
    Vec2f ret;
    Vec2f retd;
    retd.x = 2*v.x*vd.x;
    ret.x = v.x*v.x;
    retd.y = 2*v.y*vd.y;
    ret.y = v.y*v.y;
    *sq = ret;
    return retd;
}

/*
  Differentiation of sq in reverse (adjoint) mode:
   gradient     of useful results: sq.x sq.y
   with respect to varying inputs: v.x v.y
*/
void sq_b(Vec2f v, Vec2f *vb, Vec2f sqb) {
    Vec2f ret;
    Vec2f retb;
    Vec2f sq;
    ret.x = v.x*v.x;
    ret.y = v.y*v.y;
    sq = ret;
    retb = sqb;
    vb->y = 2*v.y*retb.y;
    vb->x = 2*v.x*retb.x;
}

Vec2f sq_c(Vec2f v) {
    Vec2f ret;
    ret.x = v.x*v.x;
    ret.y = v.y*v.y;
    return ret;
}

/*
  Differentiation of sub in forward (tangent) mode:
   variations   of useful results: sub.x sub.y
   with respect to varying inputs: a.x a.y b.x b.y
*/
Vec2f sub_d(Vec2f a, Vec2f ad, Vec2f b, Vec2f bd, Vec2f *sub) {
    Vec2f ret;
    Vec2f retd;
    retd.x = ad.x - bd.x;
    ret.x = a.x - b.x;
    retd.y = ad.y - bd.y;
    ret.y = a.y - b.y;
    *sub = ret;
    return retd;
}

/*
  Differentiation of sub in reverse (adjoint) mode:
   gradient     of useful results: sub.x sub.y
   with respect to varying inputs: a.x a.y b.x b.y
*/
void sub_b(Vec2f a, Vec2f *ab, Vec2f b, Vec2f *bb, Vec2f subb) {
    Vec2f ret;
    Vec2f retb;
    Vec2f sub;
    ret.x = a.x - b.x;
    ret.y = a.y - b.y;
    sub = ret;
    retb = subb;
    ab->y = retb.y;
    bb->y = -retb.y;
    ab->x = retb.x;
    bb->x = -retb.x;
}

Vec2f sub_c(Vec2f a, Vec2f b) {
    Vec2f ret;
    ret.x = a.x - b.x;
    ret.y = a.y - b.y;
    return ret;
}

/*
  Differentiation of get_vec2f_unchecked in forward (tangent) mode:
   variations   of useful results: get_vec2f_unchecked.x get_vec2f_unchecked.y
   with respect to varying inputs: *img
   Plus diff mem management of: img:in
*/
Vec2f get_vec2f_unchecked_d(int W, int H, int C, const float *img, const float
        *imgd, int i, int j, Vec2f *get_vec2f_unchecked) {
    Vec2f ret;
    Vec2f retd;
    retd.x = imgd[C*(i*H+j) + 0];
    ret.x = img[C*(i*H+j) + 0];
    retd.y = imgd[C*(i*H+j) + 1];
    ret.y = img[C*(i*H+j) + 1];
    *get_vec2f_unchecked = ret;
    return retd;
}

/*
  Differentiation of get_vec2f_unchecked in reverse (adjoint) mode:
   gradient     of useful results: get_vec2f_unchecked.x get_vec2f_unchecked.y
                *img
   with respect to varying inputs: *img
   Plus diff mem management of: img:in
*/
void get_vec2f_unchecked_b(int W, int H, int C, const float *img, float *imgb,
        int i, int j, Vec2f get_vec2f_uncheckedb) {
    Vec2f ret;
    Vec2f retb;
    Vec2f get_vec2f_unchecked;
    ret.x = img[C*(i*H+j) + 0];
    ret.y = img[C*(i*H+j) + 1];
    get_vec2f_unchecked = ret;
    retb = get_vec2f_uncheckedb;
    imgb[C*(i*H+j) + 1] = imgb[C*(i*H+j) + 1] + retb.y;
    imgb[C*(i*H+j) + 0] = imgb[C*(i*H+j) + 0] + retb.x;
}

Vec2f get_vec2f_unchecked_c(int W, int H, int C, const float *img, int i, int 
        j) {
    Vec2f ret;
    ret.x = img[C*(i*H+j) + 0];
    ret.y = img[C*(i*H+j) + 1];
    return ret;
}

/*
  Differentiation of get_vec2f in forward (tangent) mode:
   variations   of useful results: get_vec2f.x get_vec2f.y
   with respect to varying inputs: *img
   Plus diff mem management of: img:in
*/
Vec2f get_vec2f_d(int W, int H, int C, const float *img, const float *imgd, 
        int i, int j, Vec2f *get_vec2f) {
    if (i < 0 || i >= W || j < 0 || j >= H) {
        Vec2f zero;
        zero.x = 0;
        zero.y = 0;
        *get_vec2f = zero;
        return zero;
    } else {
        Vec2f ret;
        Vec2f retd;
        retd.x = imgd[C*(i*H+j) + 0];
        ret.x = img[C*(i*H+j) + 0];
        retd.y = imgd[C*(i*H+j) + 1];
        ret.y = img[C*(i*H+j) + 1];
        *get_vec2f = ret;
        return retd;
    }
}

/*
  Differentiation of get_vec2f in reverse (adjoint) mode:
   gradient     of useful results: get_vec2f.x get_vec2f.y *img
   with respect to varying inputs: *img
   Plus diff mem management of: img:in
*/
void get_vec2f_b(int W, int H, int C, const float *img, float *imgb, int i, 
        int j, Vec2f get_vec2fb) {
    Vec2f get_vec2f;
    if (i < 0 || i >= W || j < 0 || j >= H) {
        Vec2f zero;
        zero.x = 0;
        zero.y = 0;
        get_vec2f = zero;
    } else {
        Vec2f ret;
        Vec2f retb;
        ret.x = img[C*(i*H+j) + 0];
        ret.y = img[C*(i*H+j) + 1];
        get_vec2f = ret;
        retb = get_vec2fb;
        imgb[C*(i*H+j) + 1] = imgb[C*(i*H+j) + 1] + retb.y;
        imgb[C*(i*H+j) + 0] = imgb[C*(i*H+j) + 0] + retb.x;
    }
}

Vec2f get_vec2f_c(int W, int H, int C, const float *img, int i, int j) {
    if (i < 0 || i >= W || j < 0 || j >= H) {
        Vec2f zero;
        zero.x = 0;
        zero.y = 0;
        return zero;
    } else {
        Vec2f ret;
        ret.x = img[C*(i*H+j) + 0];
        ret.y = img[C*(i*H+j) + 1];
        return ret;
    }
}

bool get_mask_c(int W, int H, const bool *img, int i, int j) {
    if (i < 0 || i >= W || j < 0 || j >= H)
        return 0;
    else
        return img[i*H + j];
}

/*
  Differentiation of Rot2D in forward (tangent) mode:
   variations   of useful results: Rot2D.x Rot2D.y
   with respect to varying inputs: a
*/
Vec2f Rot2D_d(float a, float ad, Vec2f v, Vec2f *Rot2D) {
    Vec2f ret;
    Vec2f retd;
    retd.x = -((v.x*sin(a)+v.y*cos(a))*ad);
    ret.x = cos(a)*v.x - sin(a)*v.y;
    retd.y = (v.x*cos(a)-v.y*sin(a))*ad;
    ret.y = sin(a)*v.x + cos(a)*v.y;
    *Rot2D = ret;
    return retd;
}

/*
  Differentiation of Rot2D in reverse (adjoint) mode:
   gradient     of useful results: Rot2D.x Rot2D.y a
   with respect to varying inputs: a
*/
void Rot2D_b(float a, float *ab, Vec2f v, Vec2f Rot2Db) {
    Vec2f ret;
    Vec2f retb;
    Vec2f Rot2D;
    ret.x = cos(a)*v.x - sin(a)*v.y;
    ret.y = sin(a)*v.x + cos(a)*v.y;
    Rot2D = ret;
    retb = Rot2Db;
    *ab = *ab + (cos(a)*v.x-sin(a)*v.y)*retb.y - (sin(a)*v.x+cos(a)*v.y)*retb.
        x;
}

Vec2f Rot2D_c(float a, Vec2f v) {
    Vec2f ret;
    ret.x = cos(a)*v.x - sin(a)*v.y;
    ret.y = sin(a)*v.x + cos(a)*v.y;
    return ret;
}

/*
  Differentiation of reg in forward (tangent) mode:
   variations   of useful results: reg.x reg.y
   with respect to varying inputs: *Offsets_Angle
   Plus diff mem management of: Offsets_Angle:in
*/
Vec2f reg_d(int W, int H, const float *Offsets_Angle, const float *
        Offsets_Angled, const float *UrShape, const bool *Mask, int i, int j, 
        int x, int y, Vec2f *reg) {
    if (get_mask(W, H, Mask, i, j) && get_mask(W, H, Mask, i + x, j + y)) {
        Vec2f d_off;
        Vec2f d_offd;
        Vec2f result1;
        Vec2f result1d;
        int arg1;
        int arg2;
        Vec2f result2;
        Vec2f result2d;
        result1d = get_vec2f_unchecked_d(W, H, 3, Offsets_Angle, 
                                         Offsets_Angled, i, j, &result1);
        arg1 = i + x;
        arg2 = i + y;
        result2d = get_vec2f_d(W, H, 3, Offsets_Angle, Offsets_Angled, arg1, 
                               arg2, &result2);
        d_offd = sub_d(result1, result1d, result2, result2d, &d_off);
        Vec2f d_ur;
        result1 = get_vec2f_unchecked_c(W, H, 2, UrShape, i, j);
        arg1 = i + x;
        arg2 = j + y;
        result2 = get_vec2f_c(W, H, 2, UrShape, arg1, arg2);
        d_ur = sub_c(result1, result2);
        Vec2f d_diff;
        Vec2f d_diffd;
        result1d = Rot2D_d(Offsets_Angle[3*(i*H+j) + 2], Offsets_Angled[3*(i*H
                           +j) + 2], d_ur, &result1);
        d_diffd = sub_d(d_off, d_offd, result1, result1d, &d_diff);
        result1d = sq_d(d_diff, d_diffd, &result1);
        *reg = result1;
        return result1d;
    } else {
        Vec2f zero;
        zero.x = 0;
        zero.y = 0;
        *reg = zero;
        return zero;
    }
}

/*
  Differentiation of reg in reverse (adjoint) mode:
   gradient     of useful results: *Offsets_Angle reg.x reg.y
   with respect to varying inputs: *Offsets_Angle
   Plus diff mem management of: Offsets_Angle:in
*/
void reg_b(int W, int H, const float *Offsets_Angle, float *Offsets_Angleb, 
        const float *UrShape, const bool *Mask, int i, int j, int x, int y, 
        Vec2f regb) {
    Vec2f reg;
    if (get_mask(W, H, Mask, i, j) && get_mask(W, H, Mask, i + x, j + y)) {
        Vec2f d_off;
        Vec2f result1;
        int arg1;
        int arg2;
        Vec2f result2;
        result1 = get_vec2f_unchecked_c(W, H, 3, Offsets_Angle, i, j);
        arg1 = i + x;
        arg2 = i + y;
        result2 = get_vec2f_c(W, H, 3, Offsets_Angle, arg1, arg2);
        d_off = sub_c(result1, result2);
        Vec2f d_ur;
        pushReal4(result1.x);
        pushReal4(result1.y);
        result1 = get_vec2f_unchecked_c(W, H, 2, UrShape, i, j);
        pushInteger4(arg1);
        arg1 = i + x;
        pushInteger4(arg2);
        arg2 = j + y;
        pushReal4(result2.x);
        pushReal4(result2.y);
        result2 = get_vec2f_c(W, H, 2, UrShape, arg1, arg2);
        d_ur = sub_c(result1, result2);
        Vec2f d_diff;
        result1 = Rot2D_c(Offsets_Angle[3*(i*H+j) + 2], d_ur);
        d_diff = sub_c(d_off, result1);
        pushReal4(result1.x);
        pushReal4(result1.y);
        result1 = sq_c(d_diff);
        reg = result1;
        Vec2f d_offb;
        Vec2f result1b;
        Vec2f result2b;
        Vec2f d_diffb;
        result1b = regb;
        popReal4(&(result1.y));
        popReal4(&(result1.x));
        d_diffb.x = 0.0;
        d_diffb.y = 0.0;
        sq_b(d_diff, &d_diffb, result1b);
        d_offb.x = 0.0;
        d_offb.y = 0.0;
        result1b.x = 0.0;
        result1b.y = 0.0;
        sub_b(d_off, &d_offb, result1, &result1b, d_diffb);
        Rot2D_b(Offsets_Angle[3*(i*H+j) + 2], &(Offsets_Angleb[3*(i*H+j) + 2])
                , d_ur, result1b);
        popReal4(&(result2.y));
        popReal4(&(result2.x));
        popInteger4(&arg2);
        popInteger4(&arg1);
        popReal4(&(result1.y));
        popReal4(&(result1.x));
        result1b.x = 0.0;
        result1b.y = 0.0;
        result2b.x = 0.0;
        result2b.y = 0.0;
        sub_b(result1, &result1b, result2, &result2b, d_offb);
        get_vec2f_b(W, H, 3, Offsets_Angle, Offsets_Angleb, arg1, arg2, 
                    result2b);
        get_vec2f_unchecked_b(W, H, 3, Offsets_Angle, Offsets_Angleb, i, j, 
                              result1b);
    } else {
        Vec2f zero;
        zero.x = 0;
        zero.y = 0;
        reg = zero;
    }
}

Vec2f reg_c(int W, int H, const float *Offsets_Angle, const float *UrShape, 
        const bool *Mask, int i, int j, int x, int y) {
    if (get_mask(W, H, Mask, i, j) && get_mask(W, H, Mask, i + x, j + y)) {
        Vec2f d_off;
        Vec2f result1;
        int arg1;
        int arg2;
        Vec2f result2;
        result1 = get_vec2f_unchecked_c(W, H, 3, Offsets_Angle, i, j);
        arg1 = i + x;
        arg2 = i + y;
        result2 = get_vec2f_c(W, H, 3, Offsets_Angle, arg1, arg2);
        d_off = sub_c(result1, result2);
        Vec2f d_ur;
        result1 = get_vec2f_unchecked_c(W, H, 2, UrShape, i, j);
        arg1 = i + x;
        arg2 = j + y;
        result2 = get_vec2f_c(W, H, 2, UrShape, arg1, arg2);
        d_ur = sub_c(result1, result2);
        Vec2f d_diff;
        result1 = Rot2D_c(Offsets_Angle[3*(i*H+j) + 2], d_ur);
        d_diff = sub_c(d_off, result1);
        result1 = sq_c(d_diff);
        return result1;
    } else {
        Vec2f zero;
        zero.x = 0;
        zero.y = 0;
        return zero;
    }
}

/*
  Differentiation of arap in forward (tangent) mode:
   variations   of useful results: *Output
   with respect to varying inputs: *Output *Offsets_Angle
   RW status of diff variables: *Output:in-out *Offsets_Angle:in
   Plus diff mem management of: Output:in Offsets_Angle:in
*/
void arap_d(int W, int H, float w_fit, float w_reg, const float *Offsets_Angle
        , const float *Offsets_Angled, const float *UrShape, const float *
        Constraints, const bool *Mask, const bool *C_valid, int i, int j, 
        float *Output, float *Outputd) {
    /* [W, H, 3] 
 [W, H, 2] 
 [W, H, 2] 
 [W, H] 
 [W, H] 
 [10] */
    Vec2f E_fit;
    Vec2f E_fitd;
    Vec2f result1;
    Vec2f result1d;
    Vec2f result2;
    Vec2f result2d;
    Vec2f result3;
    Vec2f result3d;
    unsigned char temp;
    result1d = get_vec2f_unchecked_d(W, H, 3, Offsets_Angle, Offsets_Angled, i
                                     , j, &result1);
    result2 = get_vec2f_unchecked_c(W, H, 2, Constraints, i, j);
    result2d.x = 0.0;
    result2d.y = 0.0;
    result3d = sub_d(result1, result1d, result2, result2d, &result3);
    E_fitd = sq_d(result3, result3d, &E_fit);
    temp = (float)C_valid[i*W + j];
    Outputd[0] = temp*w_fit*E_fitd.x;
    Output[0] = temp*(w_fit*E_fit.x);
    temp = (float)C_valid[i*W + j];
    Outputd[1] = temp*w_fit*E_fitd.y;
    Output[1] = temp*(w_fit*E_fit.y);
    Vec2f reg_right;
    Vec2f reg_rightd;
    reg_rightd = reg_d(W, H, Offsets_Angle, Offsets_Angled, UrShape, Mask, i, 
                       j, 1, 0, &reg_right);
    Vec2f reg_left;
    Vec2f reg_leftd;
    reg_leftd = reg_d(W, H, Offsets_Angle, Offsets_Angled, UrShape, Mask, i, j
                      , -1, 0, &reg_left);
    Vec2f reg_bottom;
    Vec2f reg_bottomd;
    reg_bottomd = reg_d(W, H, Offsets_Angle, Offsets_Angled, UrShape, Mask, i,
                        j, 0, 1, &reg_bottom);
    Vec2f reg_top;
    Vec2f reg_topd;
    reg_topd = reg_d(W, H, Offsets_Angle, Offsets_Angled, UrShape, Mask, i, j,
                     0, -1, &reg_top);
    Outputd[2] = w_reg*reg_rightd.x;
    Output[2] = w_reg*reg_right.x;
    Outputd[3] = w_reg*reg_rightd.y;
    Output[3] = w_reg*reg_right.y;
    Outputd[4] = w_reg*reg_leftd.x;
    Output[4] = w_reg*reg_left.x;
    Outputd[5] = w_reg*reg_leftd.y;
    Output[5] = w_reg*reg_left.y;
    Outputd[6] = w_reg*reg_bottomd.x;
    Output[6] = w_reg*reg_bottom.x;
    Outputd[7] = w_reg*reg_bottomd.y;
    Output[7] = w_reg*reg_bottom.y;
    Outputd[8] = w_reg*reg_topd.x;
    Output[8] = w_reg*reg_top.x;
    Outputd[9] = w_reg*reg_topd.y;
    Output[9] = w_reg*reg_top.y;
}

/*
  Differentiation of arap in reverse (adjoint) mode:
   gradient     of useful results: *Output
   with respect to varying inputs: *Output *Offsets_Angle
   RW status of diff variables: *Output:in-out *Offsets_Angle:out
   Plus diff mem management of: Output:in Offsets_Angle:in
*/
void arap_b(int W, int H, float w_fit, float w_reg, const float *Offsets_Angle
        , float *Offsets_Angleb, const float *UrShape, const float *
        Constraints, const bool *Mask, const bool *C_valid, int i, int j, 
        float *Output, float *Outputb) {
    /* [W, H, 3] 
 [W, H, 2] 
 [W, H, 2] 
 [W, H] 
 [W, H] 
 [10] */
    Vec2f E_fit;
    Vec2f E_fitb;
    Vec2f result1;
    Vec2f result1b;
    Vec2f result2;
    Vec2f result2b;
    Vec2f result3;
    Vec2f result3b;
    result1 = get_vec2f_unchecked_c(W, H, 3, Offsets_Angle, i, j);
    result2 = get_vec2f_unchecked_c(W, H, 2, Constraints, i, j);
    result3 = sub_c(result1, result2);
    E_fit = sq_c(result3);
    Output[0] = (float)C_valid[i*W+j]*w_fit*E_fit.x;
    Output[1] = (float)C_valid[i*W+j]*w_fit*E_fit.y;
    Vec2f reg_right;
    Vec2f reg_rightb;
    reg_right = reg_c(W, H, Offsets_Angle, UrShape, Mask, i, j, 1, 0);
    Vec2f reg_left;
    Vec2f reg_leftb;
    reg_left = reg_c(W, H, Offsets_Angle, UrShape, Mask, i, j, -1, 0);
    Vec2f reg_bottom;
    Vec2f reg_bottomb;
    reg_bottom = reg_c(W, H, Offsets_Angle, UrShape, Mask, i, j, 0, 1);
    Vec2f reg_top;
    Vec2f reg_topb;
    reg_top = reg_c(W, H, Offsets_Angle, UrShape, Mask, i, j, 0, -1);
    Output[2] = w_reg*reg_right.x;
    Output[3] = w_reg*reg_right.y;
    Output[4] = w_reg*reg_left.x;
    Output[5] = w_reg*reg_left.y;
    Output[6] = w_reg*reg_bottom.x;
    Output[7] = w_reg*reg_bottom.y;
    Output[8] = w_reg*reg_top.x;
    Output[9] = w_reg*reg_top.y;
    reg_topb.y = w_reg*Outputb[9];
    Outputb[9] = 0.0;
    reg_topb.x = w_reg*Outputb[8];
    Outputb[8] = 0.0;
    reg_bottomb.y = w_reg*Outputb[7];
    Outputb[7] = 0.0;
    reg_bottomb.x = w_reg*Outputb[6];
    Outputb[6] = 0.0;
    reg_leftb.y = w_reg*Outputb[5];
    Outputb[5] = 0.0;
    reg_leftb.x = w_reg*Outputb[4];
    Outputb[4] = 0.0;
    reg_rightb.y = w_reg*Outputb[3];
    Outputb[3] = 0.0;
    reg_rightb.x = w_reg*Outputb[2];
    Outputb[2] = 0.0;
    *Offsets_Angleb = 0.0;
    reg_b(W, H, Offsets_Angle, Offsets_Angleb, UrShape, Mask, i, j, 0, -1, 
          reg_topb);
    reg_b(W, H, Offsets_Angle, Offsets_Angleb, UrShape, Mask, i, j, 0, 1, 
          reg_bottomb);
    reg_b(W, H, Offsets_Angle, Offsets_Angleb, UrShape, Mask, i, j, -1, 0, 
          reg_leftb);
    reg_b(W, H, Offsets_Angle, Offsets_Angleb, UrShape, Mask, i, j, 1, 0, 
          reg_rightb);
    E_fitb.y = w_fit*(float)C_valid[i*W+j]*Outputb[1];
    Outputb[1] = 0.0;
    E_fitb.x = w_fit*(float)C_valid[i*W+j]*Outputb[0];
    Outputb[0] = 0.0;
    result3b.x = 0.0;
    result3b.y = 0.0;
    sq_b(result3, &result3b, E_fitb);
    result1b.x = 0.0;
    result1b.y = 0.0;
    result2b.x = 0.0;
    result2b.y = 0.0;
    sub_b(result1, &result1b, result2, &result2b, result3b);
    get_vec2f_unchecked_b(W, H, 3, Offsets_Angle, Offsets_Angleb, i, j, 
                          result1b);
}
