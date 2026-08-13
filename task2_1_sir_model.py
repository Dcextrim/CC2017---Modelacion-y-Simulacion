"""
Task 2.1 - Modelo SIR (RK4) - Laboratorio 3 - Modelacion y Simulacion

Implementacion del modelo SIR usando unicamente NumPy y el metodo de
Runge-Kutta de 4to orden (RK4). Incluye:
    1. Estimacion de infectados I_hat(t) en las semanas 1-4 y calculo del SCE.
    2. Analisis de sensibilidad variando beta en +/-20% (gamma fijo).
    3. Generacion de la figura sir-sensitivity.png con las curvas I(t).
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parametros del Task 1.1
# ---------------------------------------------------------------------------
N = 500_000.0
I0 = 800.0
R0_state = 200.0
S0 = N - I0 - R0_state

beta_star = 0.40
gamma = 1.0 / 7.0  # ~0.142857

observed_data = {
    7: 1850.0,
    14: 4200.0,
    21: 8900.0,
    28: 16400.0,
}


def rk4_sir(S0, I0, R0, beta, gamma, N, days, dt=0.01):
    """Integra el sistema SIR con RK4. Devuelve arreglos t, S, I, R."""
    n_steps = int(round(days / dt))
    t = np.linspace(0.0, days, n_steps + 1)
    S = np.zeros(n_steps + 1)
    I = np.zeros(n_steps + 1)
    R = np.zeros(n_steps + 1)
    S[0], I[0], R[0] = S0, I0, R0

    def derivatives(s, i, r):
        ds = -beta * s * i / N
        di = beta * s * i / N - gamma * i
        dr = gamma * i
        return ds, di, dr

    for k in range(n_steps):
        sk, ik, rk = S[k], I[k], R[k]

        ds1, di1, dr1 = derivatives(sk, ik, rk)
        ds2, di2, dr2 = derivatives(sk + 0.5 * dt * ds1, ik + 0.5 * dt * di1, rk + 0.5 * dt * dr1)
        ds3, di3, dr3 = derivatives(sk + 0.5 * dt * ds2, ik + 0.5 * dt * di2, rk + 0.5 * dt * dr2)
        ds4, di4, dr4 = derivatives(sk + dt * ds3, ik + dt * di3, rk + dt * dr3)

        S[k + 1] = sk + (dt / 6.0) * (ds1 + 2 * ds2 + 2 * ds3 + ds4)
        I[k + 1] = ik + (dt / 6.0) * (di1 + 2 * di2 + 2 * di3 + di4)
        R[k + 1] = rk + (dt / 6.0) * (dr1 + 2 * dr2 + 2 * dr3 + dr4)

    return t, S, I, R


def part1_estimacion_y_sce():
    t, S, I, R = rk4_sir(S0, I0, R0_state, beta_star, gamma, N, days=28, dt=0.01)

    print("=" * 70)
    print("1. Estimacion semanal I_hat(t) y SCE (beta* = 0.40)")
    print("=" * 70)
    print(f"{'Semana':<8}{'Dia':<6}{'I_obs':>14}{'I_hat':>16}{'Error':>16}")

    sce = 0.0
    for week, day in enumerate([7, 14, 21, 28], start=1):
        idx = np.argmin(np.abs(t - day))
        est = I[idx]
        obs = observed_data[day]
        err = obs - est
        sce += err ** 2
        print(f"{week:<8}{day:<6}{obs:>14,.2f}{est:>16,.2f}{err:>16,.2f}")

    print("-" * 70)
    print(f"Suma de Cuadrados del Error (SCE): {sce:,.2f}")
    print()
    return sce


def part3_sensibilidad():
    print("=" * 70)
    print("3. Analisis de sensibilidad (gamma fijo, beta en {0.8, 1.0, 1.2} x beta*)")
    print("=" * 70)

    betas = {
        "-20%": 0.8 * beta_star,
        "base": 1.0 * beta_star,
        "+20%": 1.2 * beta_star,
    }

    results = {}
    days_sim = 180  # suficiente para capturar el pico y el final de la epidemia
    curves = {}
    for label, beta in betas.items():
        t, S, I, R = rk4_sir(S0, I0, R0_state, beta, gamma, N, days=days_sim, dt=0.01)
        peak_idx = np.argmax(I)
        I_peak = I[peak_idx]
        t_peak = t[peak_idx]
        final_size = R[-1] + I[-1]  # personas que pasaron por la infeccion
        pct_afectada = 100.0 * final_size / N
        R0_eff = beta / gamma
        results[label] = dict(beta=beta, R0=R0_eff, I_peak=I_peak, t_peak=t_peak,
                               final_size=final_size, pct=pct_afectada)
        curves[label] = (t, I)
        print(f"beta={beta:.2f} (R0={R0_eff:.2f}) | pico I={I_peak:,.2f} en dia {t_peak:.2f} "
              f"| tamano final={final_size:,.2f} ({pct_afectada:.2f}% de N)")

    base_peak = results["base"]["I_peak"]
    var_menos = 100.0 * (results["-20%"]["I_peak"] - base_peak) / base_peak
    var_mas = 100.0 * (results["+20%"]["I_peak"] - base_peak) / base_peak

    print("-" * 70)
    print(f"Variacion del pico vs base (-20% beta): {var_menos:+.2f}%")
    print(f"Variacion del pico vs base (+20% beta): {var_mas:+.2f}%")
    print()

    return results, curves, var_menos, var_mas


def plot_sensitivity(curves, filename="sir-sensitivity.png"):
    plt.figure(figsize=(9, 5.5))
    colors = {"-20%": "#2b8a3e", "base": "#1864ab", "+20%": "#c92a2a"}
    labels = {"-20%": "β = 0.32 (-20%)", "base": "β = 0.40 (base)", "+20%": "β = 0.48 (+20%)"}

    for key in ["-20%", "base", "+20%"]:
        t, I = curves[key]
        plt.plot(t, I, label=labels[key], color=colors[key], linewidth=2)

    plt.xlabel("Tiempo (dias)")
    plt.ylabel("Infectados activos I(t)")
    plt.title("Analisis de sensibilidad del modelo SIR ante variaciones de beta (±20%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Figura guardada en: {filename}")


if __name__ == "__main__":
    part1_estimacion_y_sce()
    results, curves, var_menos, var_mas = part3_sensibilidad()
    plot_sensitivity(curves, filename="sir-sensitivity.png")
