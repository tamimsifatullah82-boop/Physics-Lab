import numpy as np
import matplotlib.pyplot as plt

# --- Variant 12 Parameters ---
m = 3.0                         # Mass (kg)
R = 2.0                         # Radius (m)
alpha = np.pi / 2 + np.pi / 3   # Arc angular size: 5*pi/6 rad (150 deg)
mu_var = 0.02                   # Friction coefficient
g = 9.81                        # Gravity (m/s^2)
dt = 1e-4                       # Time step (s)

def d_omega(theta, omega, mu):
    return -(g / R) * np.sin(theta) - mu * (omega**2 + (g / R) * np.cos(theta))

def normal_force(theta, omega):
    return m * (R * omega**2 + g * np.cos(theta))

def passes_arc(v0, mu):
    """Returns True if the body completes the entire arc without detaching (N > 0)."""
    theta = 0.0
    omega = v0 / R
    while theta < alpha:
        if omega <= 0 or normal_force(theta, omega) <= 0:
            return False
        # RK4 step
        k1_t = omega
        k1_w = d_omega(theta, omega, mu)
        
        k2_t = omega + 0.5 * dt * k1_w
        k2_w = d_omega(theta + 0.5 * dt * k1_t, omega + 0.5 * dt * k1_w, mu)
        
        k3_t = omega + 0.5 * dt * k2_w
        k3_w = d_omega(theta + 0.5 * dt * k2_t, omega + 0.5 * dt * k2_w, mu)
        
        k4_t = omega + dt * k3_w
        k4_w = d_omega(theta + dt * k3_t, omega + dt * k3_w, mu)
        
        theta += (dt / 6.0) * (k1_t + 2*k2_t + 2*k3_t + k4_t)
        omega += (dt / 6.0) * (k1_w + 2*k2_w + 2*k3_w + k4_w)
        
    return normal_force(alpha, omega) > 0

def find_critical_v0(mu, tol=1e-5):
    """Finds minimum initial velocity v0 using the bisection method."""
    low, high = 0.1, 30.0
    while high - low > tol:
        mid = 0.5 * (low + high)
        if passes_arc(mid, mu):
            high = mid
        else:
            low = mid
    return high

def simulate_full_trajectory(v0, mu):
    """Simulates both circular arc motion and subsequent free flight."""
    # Phase 1: Motion along the ring
    theta, omega = 0.0, v0 / R
    x_arc, y_arc = [0.0], [0.0]
    
    while theta < alpha:
        k1_t = omega
        k1_w = d_omega(theta, omega, mu)
        k2_t = omega + 0.5 * dt * k1_w
        k2_w = d_omega(theta + 0.5 * dt * k1_t, omega + 0.5 * dt * k1_w, mu)
        k3_t = omega + 0.5 * dt * k2_w
        k3_w = d_omega(theta + 0.5 * dt * k2_t, omega + 0.5 * dt * k2_w, mu)
        k4_t = omega + dt * k3_w
        k4_w = d_omega(theta + dt * k3_t, omega + dt * k3_w, mu)
        
        theta += (dt / 6.0) * (k1_t + 2*k2_t + 2*k3_t + k4_t)
        omega += (dt / 6.0) * (k1_w + 2*k2_w + 2*k3_w + k4_w)
        
        if theta > alpha:
            theta = alpha
        x_arc.append(R * np.sin(theta))
        y_arc.append(R * (1.0 - np.cos(theta)))
        if normal_force(theta, omega) <= 0:
            break

    # Exit parameters
    x_exit, y_exit = x_arc[-1], y_arc[-1]
    v_exit = R * omega
    vx = v_exit * np.cos(theta)
    vy = v_exit * np.sin(theta)
    exit_angle_deg = np.degrees(np.arctan2(vy, vx))

    # Phase 2: Free ballistic flight
    x_free, y_free = [x_exit], [y_exit]
    x, y = x_exit, y_exit
    while y > 0:
        x += vx * dt
        y += vy * dt - 0.5 * g * dt**2
        vy -= g * dt
        if y < 0:
            y = 0.0
        x_free.append(x)
        y_free.append(y)

    H_max = max(y_free)
    L_free = abs(x_free[-1] - x_exit)
    return (x_arc, y_arc), (x_free, y_free), (x_exit, y_exit, vx, vy, exit_angle_deg, H_max, L_free, x_free[-1])

# --- Stage 1: Critical velocity & trajectory for Variant 12 ---
v0_crit = find_critical_v0(mu_var)
(x_arc, y_arc), (x_free, y_free), stats = simulate_full_trajectory(v0_crit, mu_var)
x_ex, y_ex, vx_ex, vy_ex, ang_ex, H_max, L_free, x_land = stats

print(f"=== Variant 12 Results (mu = {mu_var}) ===")
print(f"Critical initial velocity v0 : {v0_crit:.4f} m/s")
print(f"Exit coordinates (x, y)      : ({x_ex:.4f}, {y_ex:.4f}) m")
print(f"Exit velocity (vx, vy)       : ({vx_ex:.4f}, {vy_ex:.4f}) m/s")
print(f"Exit angle to +x axis        : {ang_ex:.2f} deg")
print(f"Max height in free flight H  : {H_max:.4f} m")
print(f"Free-flight horizontal range : {L_free:.4f} m")
print(f"Landing x-coordinate         : {x_land:.4f} m")

# Plot 1: Trajectory y(x)
plt.figure(figsize=(9, 5))
plt.plot(x_arc, y_arc, color='blue', linewidth=2.5, label='Motion along the ring (contact zone)')
plt.plot(x_free, y_free, color='red', linestyle='--', linewidth=2.2, label='Free ballistic flight')
plt.scatter([x_ex], [y_ex], color='black', zorder=5, label=f'Exit point ({x_ex:.2f}, {y_ex:.2f})')
plt.axhline(0, color='gray', linewidth=1)
plt.title(f'Trajectory y(x) for Variant 12 (v0 = {v0_crit:.3f} m/s, μ = {mu_var})')
plt.xlabel('x, m')
plt.ylabel('y, m')
plt.axis('equal')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('trajectory_var12.png', dpi=300)
plt.show()

# --- Stage 2: Effect of friction mu in [0, 0.2] with step 0.05 ---
mu_values = np.arange(0.0, 0.21, 0.05)
v0_values = [find_critical_v0(mu) for mu in mu_values]

print("\n=== Dependence v0(mu) ===")
for mu_val, v_val in zip(mu_values, v0_values):
    print(f"mu = {mu_val:.2f} -> v0 = {v_val:.4f} m/s")

# Plot 2: v0(mu)
plt.figure(figsize=(7, 4.5))
plt.plot(mu_values, v0_values, marker='o', color='darkgreen', linewidth=2)
plt.title('Critical Initial Velocity v0 vs. Friction Coefficient μ (Variant 12)')
plt.xlabel('Coefficient of friction μ')
plt.ylabel('Critical velocity v0, m/s')
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig('v0_vs_mu_var12.png', dpi=300)
plt.show()