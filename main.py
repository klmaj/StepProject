from sensor_loader import SensorDataLoader
from step_analyzer import FootSensorAnalyzer
from step_plotter import StepPlotter
from symmetry import EnhancedFootSensorAnalyzer
from symmetry_ploter import EnhancedStepPlotter

#from gait_analyzer import GaitAnalyzer
#from gait_plotter import GaitPlotter


# Przykład użycia:
if __name__ == "__main__":
    # Załaduj dane
    loader = SensorDataLoader("WybraneDaneGotowe1.xlsx")
    #loader = SensorDataLoader("Dane2025-06-06.xlsx")

    time, left, right, sheet_name = loader.load_sheet(2)
    
    # Stwórz analizator
    analyzer = FootSensorAnalyzer(time, left, right, sheet_name)
    
    # Stwórz plotter
    plotter = StepPlotter(time, analyzer)
    
    # Generuj raport
    analyzer.generate_report()
    
    # Wykresy - plotly (interaktywne)
    plotter.plot_signals()
    plotter.plot_step_detection_plotly('left')

    plotter.plot_step_periods()

    plotter.plot_pressure_heatmap_both_feet()

    plotter.plot_cross_correlation_matrix()

    step_range = (0, 10)

    # Zakres kroków, np. 5 do 10
    corr_mean = analyzer.full_correlation_matrix_by_step( step_range, method='mean')
    corr_max = analyzer.full_correlation_matrix_by_step( step_range, method='max')

    # Wywołanie wykresu
    plotter.plot_full_corr_heatmap_plotly(corr_mean, corr_max, step_range)

    left_mat, right_mat = analyzer.get_max_signal_matrix(step_range)

    # 4. Oblicz macierze SI i MAPD
    si_matrix = analyzer.full_si_map(left_mat, right_mat)
    mapd_matrix = analyzer.full_mapd_map(left_mat, right_mat)

    # 5. Narysuj podwójną heatmapę
    plotter.plot_si_map_heatmaps(si_matrix, mapd_matrix)

    plotter.plot_symmetry_indices()

    global_left, global_right = analyzer.compute_global_signals()
    mapd_time, mapd_mean = analyzer.compute_global_mapd(global_left, global_right)

    plotter.plot_global_signals(global_left, global_right)
    plotter.plot_global_mapd(mapd_time, mapd_mean)

    #symmetry_analyzer = EnhancedFootSensorAnalyzer(time, left, right, sheet_name)

   # raport = symmetry_analyzer.generate_comprehensive_report()

    # 4. Tworzenie obiektów do wizualizacji
    #plotter = EnhancedStepPlotter(symmetry_analyzer)

    # 5. Wizualizacje
    #plotter.plot_force_distribution()          # Histogramy sił
    #plotter.plot_correlation_matrix()          # Macierz korelacji czujników
    #plotter.plot_foot_correlation_analysis()   # Analiza korelacji stóp
    #plotter.plot_symmetry_analysis()           # Analiza symetrii
    #plotter.plot_cop_analysis()                # Centrum nacisku (COP)
    #plotter.plot_comprehensive_dashboard()     # Dashboard zbiorczy

# Wczytywanie danych
#loader = SensorDataLoader("WybraneDaneGotowe1.xlsx")
#time, left, right, sheet_name = loader.load_sheet(sheet_index=0)

# Analiza
#analyzer = GaitAnalyzer(time, left, right)
#analyzer.plot_step_periods_both_feet()

# Raport
#print(f"\n=== {sheet_name} ===")
#for k, (l, r, si) in summary.items():
#    print(f"{k.capitalize()}: Lewa = {l:.2f}, Prawa = {r:.2f}, SI = {si:.1f}%")

# Wykres
#plotter = GaitPlotter(time, analyzer)
#plotter.plot_signals(czas_max=20) # plot_signals(czas_max=10)
#plotter.plot_step_phases(threshold=15.0)


#plotter.plot_signal_with_phases("left", step_index=5, threshold=10)


print("hello")