from sensor_loader import SensorDataLoader
from gait_analyzer import GaitAnalyzer
from gait_plotter import GaitPlotter


# Wczytywanie danych
loader = SensorDataLoader("WybraneDaneGotowe1.xlsx")
time, left, right, sheet_name = loader.load_sheet(sheet_index=0)

# Analiza
analyzer = GaitAnalyzer(time, left, right)
#analyzer.plot_step_periods_both_feet()

# Raport
print(f"\n=== {sheet_name} ===")
#for k, (l, r, si) in summary.items():
#    print(f"{k.capitalize()}: Lewa = {l:.2f}, Prawa = {r:.2f}, SI = {si:.1f}%")

# Wykres
plotter = GaitPlotter(time, analyzer)
plotter.plot_signals(czas_max=20) # plot_signals(czas_max=10)

plotter.plot_steps_period(20,30)

print("hello")