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
    Offsets_left = np.roll(Offsets, shift=-1, axis=0)
    Offsets_right = np.roll(Offsets, shift=1, axis=0)
    Offsets_up = np.roll(Offsets, shift=-1, axis=1)
    Offsets_down = np.roll(Offsets, shift=1, axis=1)

    UrShape_left = np.roll(UrShape, shift=-1, axis=0)
    UrShape_right = np.roll(UrShape, shift=1, axis=0)
    UrShape_up = np.roll(UrShape, shift=-1, axis=1)
    UrShape_down = np.roll(UrShape, shift=1, axis=1)

    Mask_left = np.roll(Mask, shift=-1, axis=0)
    Mask_right = np.roll(Mask, shift=1, axis=0)
    Mask_up = np.roll(Mask, shift=-1, axis=1)
    Mask_down = np.roll(Mask, shift=1, axis=1)

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

# jf = f
# jJTJx = JTJx
# jJTFx = JTFx

min_fwd_time = 1e20
min_JTJ_time = 1e20
min_JTFx_time = 1e20

avg_fwd_time = 0
avg_JTJ_time = 0
avg_JTFx_time = 0

for i in range(num_iter + 1):
    start = time.time()
    y = jf(Offsets, Angle).block_until_ready()
    int0 = time.time()
    jtjx = jJTJx(Offsets, Angle, Offsets_d, Angle_d)
    jtjx[0].block_until_ready()
    jtjx[1].block_until_ready()
    int1 = time.time()
    jtfx = jJTFx(Offsets, Angle)
    jtfx[0].block_until_ready()
    jtfx[1].block_until_ready()
    end = time.time()

    if i > 0:
        avg_fwd_time += int0 - start
        avg_JTJ_time += int1 - int0
        avg_JTFx_time += end - int1
        if int0 - start < min_fwd_time:
            min_fwd_time = int0 - start
        if int1 - int0 < min_JTJ_time:
            min_JTJ_time = int1 - int0
        if end - int1 < min_JTFx_time:
            min_JTFx_time = end - int1

print('Minimum forward time:', min_fwd_time)
print('Minimum JTJ time:', min_JTJ_time)
print('Minimum JTFx time:', min_JTFx_time)
print('Ratio minimum JTJ:', min_JTJ_time / min_fwd_time)
print('Ratio minimum JTFx:', min_JTFx_time / min_fwd_time)
print('Average forward time:', avg_fwd_time / num_iter)
print('Average JTJ time:', avg_JTJ_time / num_iter)
print('Average JTFx time:', avg_JTFx_time / num_iter)
print('Ratio average JTJ:', avg_JTJ_time / avg_fwd_time)
print('Ratio average JTFx:', avg_JTFx_time / avg_fwd_time)