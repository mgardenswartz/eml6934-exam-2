import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm" 
})
DPI = 300

def my_equation(c):
    return (0.2 * c + 0.36) / (0.16 * c + 0.44)

x = np.linspace(0, 1, 100)
y = my_equation(x)

# INITIALIZATION (Must use plt to create the ax)
fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

# THE AX WORKHORSE (Use ax for everything inside the box)
ax.plot(x, y, linewidth=2.5, color='#2c3e50', label="Posterior Belief")

ax.set(
    title="",
    # Using \mathrm instead of \text for better compatibility
    xlabel=r"Prior Belief ($bel(x_{t}=\mathrm{clean})$)",
    ylabel=r"Posterior Belief ($bel(x_{t+1}=\mathrm{clean} \mid z_{t+1}=\mathrm{clean})$)",
    xlim=(0, 1),
    ylim=(0, 1),
    xticks=np.arange(0, 1.1, 0.1),
    yticks=np.arange(0, 1.1, 0.1)
)

ax.spines[['top', 'right']].set_visible(True)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# FINALIZATION (Must use plt to package and save)
plt.tight_layout()
plt.savefig('docs/q1_plot.pdf', dpi=DPI)
