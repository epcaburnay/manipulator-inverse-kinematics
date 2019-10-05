import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=1, suppress=True)


def kinematic_equations(q):
    d_1 = 300
    d_5 = 72
    a_2 = 250
    a_3 = 160
    p_inside = (d_5*np.cos(q[2])*np.sin(q[3] + np.pi/2) + d_5*np.sin(q[2])
                * np.cos(q[3] + np.pi/2) + a_3*np.cos(q[2]))*np.cos(q[1])
    p_inside -= (d_5*np.sin(q[2])*np.sin(q[3] + np.pi/2) - d_5*np.cos(q[2])
                 * np.cos(q[3] + np.pi/2) + a_3*np.sin(q[2]))*np.sin(q[1])
    p_inside += a_2*np.cos(q[1])
    p_y = p_inside*np.cos(q[0])
    p_x = p_inside*np.sin(q[0])
    p_z = (d_5*np.cos(q[2])*np.sin(q[3] + np.pi/2) + d_5*np.sin(q[2])
           * np.cos(q[3] + np.pi/2) + a_3*np.cos(q[2]))*np.sin(q[1])
    p_z += (d_5*np.sin(q[2])*np.sin(q[3] + np.pi/2) - d_5*np.cos(q[2])
            * np.cos(q[3] + np.pi/2) + a_3*np.sin(q[2]))*np.cos(q[1])
    p_z += a_2*np.sin(q[1]) + d_1
    return np.array([-p_x, p_y, p_z])


def objective(q, px, py, pz):
    """Function to be minimized"""
    p = kinematic_equations(q)
    return (p[1] - py)**2 + (p[0] - px)**2 + (p[2] - pz)**2


b_1 = np.deg2rad((-150, 150))
b_2 = np.deg2rad((-30, 100))
b_3 = np.deg2rad((-110, 0))
b_4 = np.deg2rad((-90, 90))
bnds = (b_1, b_2, b_3, b_4)

q0 = np.array(np.deg2rad([-150, -30, -110, -90]))
r = np.array(np.deg2rad([45, 45, -45, 45]))
p = np.array(kinematic_equations(r))
print(p)
sol = minimize(objective, q0, args=(
    p[0], p[1], p[2]), method='SLSQP', bounds=bnds)
print(sol)
print(np.rad2deg(sol.x))
