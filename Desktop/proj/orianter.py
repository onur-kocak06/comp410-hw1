import math

def rotate_y(x, y, z, theta_deg):
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_new = x * cos_t + z * sin_t
    z_new = -x * sin_t + z * cos_t
    return x_new, y, z_new
def rotate_x(x, y, z, theta_deg):
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    y_new = y * cos_t - z * sin_t
    z_new = y * sin_t + z * cos_t
    return x, y_new, z_new
def rotate_z(x, y, z, theta_deg):
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_new = x * cos_t - y * sin_t
    y_new = x * sin_t + y * cos_t
    return x_new, y_new, z

with open("model.obj", "r") as f:
    lines = f.readlines()

theta_degy = -25
theta_degx = -25
theta_degz = -15

with open("model_rotated.obj", "w") as f:
    for line in lines:
        if line.startswith("v "):
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            x, y, z = rotate_y(x, y, z, theta_degy)
            x, y, z = rotate_x(x, y, z, theta_degx)
            x, y, z = rotate_z(x, y, z, theta_degz)
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        else:
            f.write(line)
