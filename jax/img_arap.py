import jax
import jax.numpy as np
import time
import skimage.io

num_iter = 10

key = jax.random.PRNGKey(1234)
Mask = np.array(skimage.io.imread('../data/Mask0.png')) > 0
Mask = np.reshape(Mask, [Mask.shape[0], Mask.shape[1], 1])
Offsets = jax.random.uniform(key, shape=[Mask.shape[0], Mask.shape[1], 2], dtype=np.float32)
Angle = jax.random.uniform(key, shape=[Mask.shape[0], Mask.shape[1]], dtype=np.float32)
Offsets_d = jax.random.uniform(key, shape=[Mask.shape[0], Mask.shape[1], 2], dtype=np.float32)
Angle_d = jax.random.uniform(key, shape=[Mask.shape[0], Mask.shape[1]], dtype=np.float32)
UrShape = jax.random.uniform(key, shape=[Mask.shape[0], Mask.shape[1], 2], dtype=np.float32)
Constraints = jax.random.uniform(key, shape=[Mask.shape[0], Mask.shape[1], 2], dtype=np.float32)
C_valid = np.ones(Mask.shape, dtype=Mask.dtype)

def f(Offsets, Angle):
    Offsets_left = np.roll(Offsets, 0, -1)
    Offsets_right = np.roll(Offsets, 0, 1)
    Offsets_up = np.roll(Offsets, 1, -1)
    Offsets_down = np.roll(Offsets, 1, 1)

    UrShape_left = np.roll(UrShape, 0, -1)
    UrShape_right = np.roll(UrShape, 0, 1)
    UrShape_up = np.roll(UrShape, 1, -1)
    UrShape_down = np.roll(UrShape, 1, 1)

    Mask_left = np.roll(Mask, 0, -1)
    Mask_right = np.roll(Mask, 0, 1)
    Mask_up = np.roll(Mask, 1, -1)
    Mask_down = np.roll(Mask, 1, 1)

    d_off_left = Offsets - Offsets_left
    d_off_right = Offsets - Offsets_right
    d_off_up = Offsets - Offsets_up
    d_off_down = Offsets - Offsets_down

    d_ur_left = UrShape - UrShape_left
    d_ur_right = UrShape - UrShape_right
    d_ur_up = UrShape - UrShape_up
    d_ur_down = UrShape - UrShape_down

    cos_angle = np.cos(Angle)
    sin_angle = np.sin(Angle)

    Rot2D_left = np.stack(\
        [cos_angle * d_ur_left[:, :, 0] - sin_angle * d_ur_left[:, :, 1],
         sin_angle * d_ur_left[:, :, 0] - cos_angle * d_ur_left[:, :, 1]], -1)
    Rot2D_right = np.stack(\
        [cos_angle * d_ur_right[:, :, 0] - sin_angle * d_ur_right[:, :, 1],
         sin_angle * d_ur_right[:, :, 0] - cos_angle * d_ur_right[:, :, 1]], -1)
    Rot2D_up = np.stack(\
        [cos_angle * d_ur_up[:, :, 0] - sin_angle * d_ur_up[:, :, 1],
         sin_angle * d_ur_up[:, :, 0] - cos_angle * d_ur_up[:, :, 1]], -1)
    Rot2D_down = np.stack(\
        [cos_angle * d_ur_down[:, :, 0] - sin_angle * d_ur_down[:, :, 1],
         sin_angle * d_ur_down[:, :, 0] - cos_angle * d_ur_down[:, :, 1]], -1)

    d_diff_left = d_off_left - Rot2D_left
    d_diff_right = d_off_right - Rot2D_right
    d_diff_up = d_off_up - Rot2D_up
    d_diff_down = d_off_down - Rot2D_down

    reg_left = np.logical_and(Mask, Mask_left) * d_diff_left * d_diff_left
    reg_right = np.logical_and(Mask, Mask_right) * d_diff_right * d_diff_right
    reg_up = np.logical_and(Mask, Mask_up) * d_diff_up * d_diff_up
    reg_down = np.logical_and(Mask, Mask_down) * d_diff_down * d_diff_down

    E_fit = (Offsets - Constraints) * (Offsets - Constraints)
    return np.stack([C_valid * 0.5 * E_fit,
                     0.5 * reg_left,
                     0.5 * reg_right,
                     0.5 * reg_up,
                     0.5 * reg_down], -1)

def JTJx(Offsets, Angle, Offsets_d, Angle_d):
    _, Jx = jax.jvp(f, [Offsets, Angle], [Offsets_d, Angle_d])
    _, f_vjp = jax.vjp(f, Offsets, Angle)
    return f_vjp(Jx)

def JTFx(Offsets, Angle):
    Fx, f_vjp = jax.vjp(f, Offsets, Angle)
    return f_vjp(Fx)

jf = jax.jit(f)
jJTJx = jax.jit(JTJx)
jJTFx = jax.jit(JTFx)

fwd_time = 1e20
JTJ_time = 1e20
JTFx_time = 1e20

for i in range(num_iter):
    start = time.time()
    y = jf(Offsets, Angle)
    int0 = time.time()
    jtjx = jJTJx(Offsets, Angle, Offsets_d, Angle_d)
    int1 = time.time()
    jtfx = jJTFx(Offsets, Angle)
    end = time.time()

    if int0 - start < fwd_time:
        fwd_time = int0 - start
    if int1 - int0 < JTJ_time:
        JTJ_time = int1 - int0
    if end - int1 < JTFx_time:
        JTFx_time = end - int1

print('Minimum forward time:', fwd_time)
print('Minimum JTJ time:', JTJ_time)
print('Minimum JTFx time:', JTFx_time)
print('Ratio JTJ:', JTJ_time / fwd_time)
print('Ratio JTFx:', JTFx_time / fwd_time)