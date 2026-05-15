import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

print('Running Cantera version:', ct.__version__)

def ignition_delay(states, species='OH'):
    y = states(species).Y
    i_ign = y.argmax()
    return states.t[i_ign]

pressures_atm = [2, 10, 20]

temperature_ranges = {
    2:  np.linspace(890, 1250, 35),
    10: np.linspace(960, 1280, 35),
    20: np.linspace(1000, 1280, 35)
}

colors = ['blue', 'green', 'red']

gas = ct.Solution('optimized_syngas.yaml')


# COMBINED FIGURE

fig_combined, ax_combined = plt.subplots(figsize=(7,6))


# LOOP OVER PRESSURES

for P_atm, color in zip(pressures_atm, colors):

    temperatures = temperature_ranges[P_atm]
    reactor_pressure = P_atm * 101325

    tau = []

    for T in temperatures:

        gas.TP = T, reactor_pressure

        gas.X = {
            'H2': 3.5,
            'CO': 3.5,
            'O2': 3.5,
            'CO2': 3.0,
            'AR': 211.5
        }

        r = ct.Reactor(gas)
        sim = ct.ReactorNet([r])

        states = ct.SolutionArray(gas, extra=['t'])

        t = 0
        max_time = 2e-2

        while t < max_time:
            t = sim.step()
            states.append(r.thermo.state, t=t)

        tau_i = ignition_delay(states, 'OH')
        tau.append(tau_i)

    tau_microseconds = np.array(tau) * 1e6

    
    # INDIVIDUAL FIGURE
    
    fig, ax = plt.subplots(figsize=(6,5))

    ax.semilogy(1000 / temperatures,
                tau_microseconds,
                color=color,
                linewidth=2)

    ax.set_xlabel('1000 / T (1/K)')
    ax.set_ylabel('Ignition delay time (μs)')
    ax.set_title(f'Ignition Delay at {P_atm} atm')
    ax.grid(True)

    # SAVE INDIVIDUAL GRAPH
    fig.savefig(f'IgnitionDelay_{P_atm}atm.png',
                dpi=300,
                bbox_inches='tight')

    
    # ADD TO COMBINED GRAPH
    
    ax_combined.semilogy(1000 / temperatures,
                         tau_microseconds,
                         color=color,
                         linewidth=2,
                         label=f'{P_atm} atm')


# FORMAT COMBINED GRAPH

ax_combined.set_xlabel('1000 / T (1/K)')
ax_combined.set_ylabel('Ignition delay time (μs)')
ax_combined.set_title('Combined Ignition Delay Curves')
ax_combined.legend()
ax_combined.grid(True)

# SAVE COMBINED GRAPH
fig_combined.savefig('Combined_IgnitionDelay.png',
                     dpi=300,
                     bbox_inches='tight')

# SHOW ALL FIGURES
plt.show()