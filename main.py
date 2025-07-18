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
    #loader = SensorDataLoader("WybraneDaneGotowe1.xlsx")
    loader = SensorDataLoader("Dane2025-06-06.xlsx")

    time, left, right, sheet_name = loader.load_sheet(5)
    
    # Stwórz analizator
    analyzer = FootSensorAnalyzer(time, left, right, sheet_name)
    
    # Stwórz plotter
    plotter = StepPlotter(time, analyzer)
    
    # Generuj raport
    analyzer.generate_report()
    
    # Wykresy - plotly (interaktywne)
    plotter.plot_signals()
    plotter.plot_step_detection_plotly('left')
    plotter.plot_step_detection_plotly('right')

    plotter.plot_step_periods()

    plotter.plot_pressure_heatmap_both_feet()

    plotter.plot_cross_correlation_matrix()

    # Zakres kroków, np. 5 do 10
    step_range = (0, 10)
    
    corr_mean = analyzer.full_correlation_matrix_by_step( step_range, method='mean')
    corr_max = analyzer.full_correlation_matrix_by_step( step_range, method='max')

    plotter.plot_full_corr_heatmap_plotly(corr_mean, corr_max, step_range)

    left_mat, right_mat = analyzer.get_max_signal_matrix(step_range)

    # Oblicza macierze SI i MAPD
    si_matrix = analyzer.full_si_map(left_mat, right_mat)
    mapd_matrix = analyzer.full_mapd_map(left_mat, right_mat)

    # Rysuj podwójną heatmapę
    plotter.plot_si_map_heatmaps(si_matrix, mapd_matrix)

    plotter.plot_symmetry_indices()

    global_left, global_right = analyzer.compute_global_signals()
    mapd_time, mapd_mean = analyzer.compute_global_mapd(global_left, global_right)

    plotter.plot_global_signals(global_left, global_right)
    

    