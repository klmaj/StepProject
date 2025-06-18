import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

class EnhancedStepPlotter:
    def __init__(self, analyzer):
        """
        Inicjalizuje plotter z analizatorem
        
        Args:
            analyzer: instancja klasy EnhancedFootSensorAnalyzer
        """
        self.analyzer = analyzer
        self.fig_size = (15, 10)
        
        # Kolory dla wizualizacji
        self.colors = {
            'left': '#2E86AB',      # Niebieski
            'right': '#A23B72',     # Różowy
            'both': '#F18F01',      # Pomarańczowy
            'background': '#F5F5F5'
        }
        
        # Style wykresów
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_force_distribution(self, save_path=None):
        """Rysuje rozkład sił dla wszystkich czujników"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Rozkład sił na czujnikach - {self.analyzer.sheet_name}', fontsize=16, fontweight='bold')
        
        for i, sensor_name in enumerate(self.analyzer.sensor_names):
            row = i // 4
            col = i % 4
            
            left_data = self.analyzer.left[:, i]
            right_data = self.analyzer.right[:, i]
            
            # Histogram
            axes[row, col].hist(left_data, bins=30, alpha=0.7, label='Lewa', 
                               color=self.colors['left'], density=True)
            axes[row, col].hist(right_data, bins=30, alpha=0.7, label='Prawa', 
                               color=self.colors['right'], density=True)
            
            axes[row, col].set_title(f'Czujnik {sensor_name}')
            axes[row, col].set_xlabel('Siła')
            axes[row, col].set_ylabel('Gęstość')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, save_path=None):
        """Rysuje macierz korelacji między czujnikami"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Macierz korelacji dla lewej stopy
        left_corr = np.corrcoef(self.analyzer.left.T)
        im1 = ax1.imshow(left_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_title('Korelacje między czujnikami - Lewa stopa', fontweight='bold')
        ax1.set_xticks(range(8))
        ax1.set_yticks(range(8))
        ax1.set_xticklabels(self.analyzer.sensor_names)
        ax1.set_yticklabels(self.analyzer.sensor_names)
        
        # Dodaj wartości do macierzy
        for i in range(8):
            for j in range(8):
                ax1.text(j, i, f'{left_corr[i, j]:.2f}', 
                        ha="center", va="center", color='black', fontsize=8)
        
        # Macierz korelacji dla prawej stopy
        right_corr = np.corrcoef(self.analyzer.right.T)
        im2 = ax2.imshow(right_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_title('Korelacje między czujnikami - Prawa stopa', fontweight='bold')
        ax2.set_xticks(range(8))
        ax2.set_yticks(range(8))
        ax2.set_xticklabels(self.analyzer.sensor_names)
        ax2.set_yticklabels(self.analyzer.sensor_names)
        
        for i in range(8):
            for j in range(8):
                ax2.text(j, i, f'{right_corr[i, j]:.2f}', 
                        ha="center", va="center", color='black', fontsize=8)
        
        # Wspólny colorbar
        cbar = fig.colorbar(im1, ax=[ax1, ax2], shrink=0.8)
        cbar.set_label('Współczynnik korelacji', rotation=270, labelpad=20)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_foot_correlation_analysis(self, save_path=None):
        """Wizualizuje analizę korelacji między stopami"""
        correlation_results = self.analyzer.analyze_foot_correlation()
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. Korelacje dla poszczególnych czujników
        ax1 = fig.add_subplot(gs[0, :2])
        sensors = list(correlation_results['sensor_correlations'].keys())
        correlations = [correlation_results['sensor_correlations'][s]['correlation'] for s in sensors]
        p_values = [correlation_results['sensor_correlations'][s]['p_value'] for s in sensors]
        
        bars = ax1.bar(sensors, correlations, color=[self.colors['left'] if p < 0.05 else 'lightgray' for p in p_values])
        ax1.set_title('Korelacje między stopami dla poszczególnych czujników', fontweight='bold')
        ax1.set_ylabel('Współczynnik korelacji')
        ax1.set_xlabel('Czujnik')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Dodaj znaczniki istotności
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.05:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        '*', ha='center', va='bottom', fontsize=16, color='red')
        
        # 2. Korelacja sił całkowitych w czasie
        ax2 = fig.add_subplot(gs[0, 2])
        left_total = np.sum(self.analyzer.left, axis=1)
        right_total = np.sum(self.analyzer.right, axis=1)
        
        ax2.scatter(left_total, right_total, alpha=0.6, c=self.colors['both'])
        ax2.set_xlabel('Siła całkowita - lewa stopa')
        ax2.set_ylabel('Siła całkowita - prawa stopa')
        ax2.set_title('Korelacja sił całkowitych')
        
        # Dodaj linię trendu
        z = np.polyfit(left_total, right_total, 1)
        p = np.poly1d(z)
        ax2.plot(left_total, p(left_total), "r--", alpha=0.8)
        
        corr_text = f"r = {correlation_results['total_force_correlation']['correlation']:.3f}"
        ax2.text(0.05, 0.95, corr_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # 3. Cross-correlation
        if 'temporal_correlations' in correlation_results:
            ax3 = fig.add_subplot(gs[1, :])
            temp_corr = correlation_results['temporal_correlations']
            
            # Ogranic zakres dla lepszej wizualizacji
            max_lag_samples = min(len(temp_corr['lags'])//4, 100)
            center_idx = len(temp_corr['lags']) // 2
            
            lags_subset = temp_corr['lags'][center_idx-max_lag_samples:center_idx+max_lag_samples]
            corr_subset = temp_corr['cross_correlation'][center_idx-max_lag_samples:center_idx+max_lag_samples]
            
            # Przelicz lagi na czas
            lags_time = lags_subset / self.analyzer.sampling_rate
            
            ax3.plot(lags_time, corr_subset, color=self.colors['both'], linewidth=2)
            ax3.axvline(x=temp_corr['optimal_lag_time'], color='red', linestyle='--', 
                       label=f'Optymalne opóźnienie: {temp_corr["optimal_lag_time"]:.3f}s')
            ax3.set_xlabel('Opóźnienie [s]')
            ax3.set_ylabel('Korelacja krzyżowa')
            ax3.set_title('Analiza korelacji krzyżowej (synchronizacja stóp)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. COP korelacje
        if 'cop_correlations' in correlation_results and correlation_results['cop_correlations']:
            ax4 = fig.add_subplot(gs[2, 0])
            cop_corr = correlation_results['cop_correlations']
            directions = ['X', 'Y']
            values = [cop_corr['x_direction']['correlation'], cop_corr['y_direction']['correlation']]
            
            bars = ax4.bar(directions, values, color=[self.colors['left'], self.colors['right']])
            ax4.set_title('Korelacje COP')
            ax4.set_ylabel('Współczynnik korelacji')
            ax4.set_ylim(-1, 1)
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Analiza korelacji między stopami - {self.analyzer.sheet_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_symmetry_analysis(self, save_path=None):
        """Wizualizuje analizę symetrii"""
        symmetry_results = self.analyzer.analyze_symmetry()
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # 1. Symetria sił dla czujników
        if 'force_symmetry' in symmetry_results:
            ax1 = fig.add_subplot(gs[0, 0])
            force_sym = symmetry_results['force_symmetry']
            
            if 'sensor_symmetries' in force_sym:
                sensors = list(force_sym['sensor_symmetries'].keys())
                values = list(force_sym['sensor_symmetries'].values())
                
                bars = ax1.bar(sensors, values, color=self.colors['both'])
                ax1.set_title('Asymetria sił dla czujników', fontweight='bold')
                ax1.set_ylabel('Wskaźnik asymetrii [%]')
                ax1.set_xlabel('Czujnik')
                
                # Linie progowe
                ax1.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Dobra symetria')
                ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Umiarkowana')
                ax1.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Znacząca asymetria')
                ax1.legend()
        
        # 2. Wykres radarowy symetrii
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        
        # Zbierz wskaźniki
        categories = []
        values = []
        
        if 'force_symmetry' in symmetry_results:
            categories.append('Siły')
            values.append(min(symmetry_results['force_symmetry']['total_force_symmetry_index'], 50))
        
        if 'temporal_symmetry' in symmetry_results and 'temporal_symmetry_index' in symmetry_results['temporal_symmetry']:
            categories.append('Czasowa')
            values.append(min(symmetry_results['temporal_symmetry']['temporal_symmetry_index'], 50))
        
        if 'spatial_symmetry' in symmetry_results:
            categories.append('Przestrzenna')
            values.append(min(symmetry_results['spatial_symmetry']['mean_spatial_asymmetry'], 50))
        
        if categories and values:
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Zamknij wykres
            angles += angles[:1]
            
            ax2.plot(angles, values, color=self.colors['both'], linewidth=2, marker='o')
            ax2.fill(angles, values, color=self.colors['both'], alpha=0.25)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories)
            ax2.set_ylim(0, 50)
            ax2.set_title('Profil asymetrii', fontweight='bold', pad=20)
        
        # 3. Porównanie średnich sił
        ax3 = fig.add_subplot(gs[0, 2])
        left_means = np.mean(self.analyzer.left, axis=0)
        right_means = np.mean(self.analyzer.right, axis=0)
        
        x = np.arange(len(self.analyzer.sensor_names))
        width = 0.35
        
        ax3.bar(x - width/2, left_means, width, label='Lewa', color=self.colors['left'])
        ax3.bar(x + width/2, right_means, width, label='Prawa', color=self.colors['right'])
        
        ax3.set_xlabel('Czujnik')
        ax3.set_ylabel('Średnia siła')
        ax3.set_title('Porównanie średnich sił')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.analyzer.sensor_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Analiza kroków
        ax4 = fig.add_subplot(gs[1, :2])
        left_steps, left_total = self.analyzer.detect_steps('left')
        right_steps, right_total = self.analyzer.detect_steps('right')
        
        ax4.plot(self.analyzer.time, left_total, label='Lewa stopa', color=self.colors['left'])
        ax4.plot(self.analyzer.time, right_total, label='Prawa stopa', color=self.colors['right'])
        
        # Zaznacz kroki
        for start, end in left_steps:
            ax4.axvspan(self.analyzer.time[start], self.analyzer.time[end], 
                       alpha=0.3, color=self.colors['left'])
        
        for start, end in right_steps:
            ax4.axvspan(self.analyzer.time[start], self.analyzer.time[end], 
                       alpha=0.3, color=self.colors['right'])
        
        ax4.set_xlabel('Czas [s]')
        ax4.set_ylabel('Siła całkowita')
        ax4.set_title('Wykrywanie kroków')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Podsumowanie numeryczne
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        summary_text = []
        if 'overall_asymmetry_index' in symmetry_results:
            summary_text.append(f"Ogólny wskaźnik asymetrii:")
            summary_text.append(f"{symmetry_results['overall_asymmetry_index']:.1f}%")
            summary_text.append("")
        
        if 'force_symmetry' in symmetry_results:
            summary_text.append(f"Asymetria sił: {symmetry_results['force_symmetry']['total_force_symmetry_index']:.1f}%")
        
        if 'temporal_symmetry' in symmetry_results and 'temporal_symmetry_index' in symmetry_results['temporal_symmetry']:
            summary_text.append(f"Asymetria czasowa: {symmetry_results['temporal_symmetry']['temporal_symmetry_index']:.1f}%")
        
        summary_text.append("")
        summary_text.append(f"Liczba kroków:")
        summary_text.append(f"Lewa: {len(left_steps)}")
        summary_text.append(f"Prawa: {len(right_steps)}")
        
        ax5.text(0.1, 0.9, '\n'.join(summary_text), transform=ax5.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background']))
        
        plt.suptitle(f'Analiza symetrii - {self.analyzer.sheet_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cop_analysis(self, save_path=None):
        """Wizualizuje analizę centrum nacisku (COP)"""
        cop_data = self.analyzer.calculate_cop_trajectory('both')
        
        if 'left' not in cop_data or 'right' not in cop_data:
            print("Nie można wygenerować wykresów COP - brak danych")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # 1. Trajektorie COP
        ax1 = fig.add_subplot(gs[0, 0])
        
        left_cop_x = cop_data['left']['cop_x']
        left_cop_y = cop_data['left']['cop_y']
        right_cop_x = cop_data['right']['cop_x']
        right_cop_y = cop_data['right']['cop_y']
        
        # Usuń NaN dla wizualizacji
        left_valid = ~(np.isnan(left_cop_x) | np.isnan(left_cop_y))
        right_valid = ~(np.isnan(right_cop_x) | np.isnan(right_cop_y))
        
        if np.sum(left_valid) > 0:
            ax1.plot(left_cop_x[left_valid], left_cop_y[left_valid], 
                    color=self.colors['left'], label='Lewa', alpha=0.7)
        
        if np.sum(right_valid) > 0:
            ax1.plot(right_cop_x[right_valid], right_cop_y[right_valid], 
                    color=self.colors['right'], label='Prawa', alpha=0.7)
        
        # Dodaj kontury stopy
        self._add_foot_outline(ax1)
        ax1.set_xlabel('Pozycja X [mm]')
        ax1.set_ylabel('Pozycja Y [mm]')
        ax1.set_title('Trajektorie COP')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 2. COP w czasie - kierunek X
        ax2 = fig.add_subplot(gs[0, 1])
        valid_time_left = self.analyzer.time[left_valid]
        valid_time_right = self.analyzer.time[right_valid]
        
        if len(valid_time_left) > 0:
            ax2.plot(valid_time_left, left_cop_x[left_valid], 
                    color=self.colors['left'], label='Lewa')
        
        if len(valid_time_right) > 0:
            ax2.plot(valid_time_right, right_cop_x[right_valid], 
                    color=self.colors['right'], label='Prawa')
        
        ax2.set_xlabel('Czas [s]')
        ax2.set_ylabel('COP X [mm]')
        ax2.set_title('COP w kierunku przód-tył')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. COP w czasie - kierunek Y
        ax3 = fig.add_subplot(gs[0, 2])
        
        if len(valid_time_left) > 0:
            ax3.plot(valid_time_left, left_cop_y[left_valid], 
                    color=self.colors['left'], label='Lewa')
        
        if len(valid_time_right) > 0:
            ax3.plot(valid_time_right, right_cop_y[right_valid], 
                    color=self.colors['right'], label='Prawa')
        
        ax3.set_xlabel('Czas [s]')
        ax3.set_ylabel('COP Y [mm]')
        ax3.set_title('COP w kierunku przyśrodkowo-bocznym')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Porównanie parametrów COP
        ax4 = fig.add_subplot(gs[1, 0])
        
        parameters = ['Długość ścieżki\n[mm]', 'Obszar wychwiań\n[mm²]']
        left_values = [cop_data['left']['path_length'], cop_data['left']['sway_area']]
        right_values = [cop_data['right']['path_length'], cop_data['right']['sway_area']]
        
        x = np.arange(len(parameters))
        width = 0.35
        
        ax4.bar(x - width/2, left_values, width, label='Lewa', color=self.colors['left'])
        ax4.bar(x + width/2, right_values, width, label='Prawa', color=self.colors['right'])
        
        ax4.set_ylabel('Wartość')
        ax4.set_title('Porównanie parametrów COP')
        ax4.set_xticks(x)
        ax4.set_xticklabels(parameters)
        ax4.legend()
        
        # 5. Histogram prędkości COP
        ax5 = fig.add_subplot(gs[1, 1])
        
        left_velocity = cop_data['left']['velocity']
        right_velocity = cop_data['right']['velocity']
        
        if len(left_velocity) > 0:
            ax5.hist(left_velocity, bins=30, alpha=0.7, label='Lewa', 
                    color=self.colors['left'], density=True)
        
        if len(right_velocity) > 0:
            ax5.hist(right_velocity, bins=30, alpha=0.7, label='Prawa', 
                    color=self.colors['right'], density=True)
        
        ax5.set_xlabel('Prędkość COP [mm/s]')
        ax5.set_ylabel('Gęstość')
        ax5.set_title('Rozkład prędkości COP')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Statystyki
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        stats_text = []
        stats_text.append("Statystyki COP:")
        stats_text.append("")
        stats_text.append("LEWA STOPA:")
        stats_text.append(f"Długość ścieżki: {cop_data['left']['path_length']:.1f} mm")
        stats_text.append(f"Obszar wychwiań: {cop_data['left']['sway_area']:.1f} mm²")
        if len(left_velocity) > 0:
            stats_text.append(f"Śr. prędkość: {np.mean(left_velocity):.1f} mm/s")
        stats_text.append("")
        stats_text.append("PRAWA STOPA:")
        stats_text.append(f"Długość ścieżki: {cop_data['right']['path_length']:.1f} mm")
        stats_text.append(f"Obszar wychwiań: {cop_data['right']['sway_area']:.1f} mm²")
        if len(right_velocity) > 0:
            stats_text.append(f"Śr. prędkość: {np.mean(right_velocity):.1f} mm/s")
        
        ax6.text(0.1, 0.9, '\n'.join(stats_text), transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background']))
        
        plt.suptitle(f'Analiza centrum nacisku (COP) - {self.analyzer.sheet_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _add_foot_outline(self, ax):
        """Dodaje kontur stopy do wykresu"""
        # Pozycje czujników
        sensor_pos = self.analyzer.sensor_positions
        
        # Rysuj czujniki
        for i, pos in enumerate(sensor_pos):
            ax.scatter(pos[0], pos[1], s=100, c='gray', alpha=0.5, marker='s')
            ax.annotate(str(i+1), (pos[0], pos[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        # Uproszczony kontur stopy
        foot_outline = np.array([
            [-10, -40], [200, -40], [200, 40], [-10, 40], [-10, -40]
        ])
        ax.plot(foot_outline[:, 0], foot_outline[:, 1], 'k--', alpha=0.3)
    
    def plot_comprehensive_dashboard(self, save_path=None):
        """Tworzy kompleksowy dashboard z wszystkimi analizami"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Sygnały w czasie
        ax1 = fig.add_subplot(gs[0, :2])
        left_total = np.sum(self.analyzer.left, axis=1)
        right_total = np.sum(self.analyzer.right, axis=1)
        
        ax1.plot(self.analyzer.time, left_total, label='Lewa', color=self.colors['left'])
        ax1.plot(self.analyzer.time, right_total, label='Prawa', color=self.colors['right'])
        ax1.set_title('Sygnały sił całkowitych w czasie')
        ax1.set_xlabel('Czas [s]')
        ax1.set_ylabel('Siła całkowita')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Korelacja między stopami
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.scatter(left_total, right_total, alpha=0.6, c=self.colors['both'])
        ax2.set_xlabel('Lewa stopa')
        ax2.set_ylabel('Prawa stopa')
        ax2.set_title('Korelacja sił całkowitych')
        
        # Linia trendu
        z = np.polyfit(left_total, right_total, 1)
        p = np.poly1d(z)
        ax2.plot(left_total, p(left_total), "r--", alpha=0.8)
        
        # Dodaj współczynnik korelacji
        from scipy.stats import pearsonr
        corr, _ = pearsonr(left_total, right_total)
        ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        ax2.grid(True, alpha=0.3)
        
        # 3. Mapa ciepła czujników - lewa stopa
        ax3 = fig.add_subplot(gs[1, 0])
        left_means = np.mean(self.analyzer.left, axis=0).reshape(2, 4)
        im3 = ax3.imshow(left_means, cmap='Reds', aspect='auto')
        ax3.set_title('Mapa sił - Lewa')
        ax3.set_xticks(range(4))
        ax3.set_yticks(range(2))
        ax3.set_xticklabels(['1', '3', '5', '7'])
        ax3.set_yticklabels(['Przyśr.', 'Boczna'])
        
        # Dodaj wartości do mapy ciepła
        for i in range(2):
            for j in range(4):
                ax3.text(j, i, f'{left_means[i, j]:.0f}', 
                        ha="center", va="center", color='white', fontweight='bold')
        
        # 4. Mapa ciepła czujników - prawa stopa
        ax4 = fig.add_subplot(gs[1, 1])
        right_means = np.mean(self.analyzer.right, axis=0).reshape(2, 4)
        im4 = ax4.imshow(right_means, cmap='Blues', aspect='auto')
        ax4.set_title('Mapa sił - Prawa')
        ax4.set_xticks(range(4))
        ax4.set_yticks(range(2))
        ax4.set_xticklabels(['2', '4', '6', '8'])
        ax4.set_yticklabels(['Przyśr.', 'Boczna'])
        
        # Dodaj wartości do mapy ciepła
        for i in range(2):
            for j in range(4):
                ax4.text(j, i, f'{right_means[i, j]:.0f}', 
                        ha="center", va="center", color='white', fontweight='bold')
        
        # 5. Wykres symetrii
        ax5 = fig.add_subplot(gs[1, 2:])
        symmetry_results = self.analyzer.analyze_symmetry()
        
        x = np.arange(len(self.analyzer.sensor_names))
        width = 0.35
        left_sensor_means = np.mean(self.analyzer.left, axis=0)
        right_sensor_means = np.mean(self.analyzer.right, axis=0)
        
        ax5.bar(x - width/2, left_sensor_means, width, label='Lewa', color=self.colors['left'])
        ax5.bar(x + width/2, right_sensor_means, width, label='Prawa', color=self.colors['right'])
        ax5.set_xlabel('Czujnik')
        ax5.set_ylabel('Średnia siła')
        ax5.set_title('Porównanie średnich sił czujników')
        ax5.set_xticks(x)
        ax5.set_xticklabels(self.analyzer.sensor_names)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Trajektorie COP
        ax6 = fig.add_subplot(gs[2, :2])
        cop_data = self.analyzer.calculate_cop_trajectory('both')
        
        if 'left' in cop_data and 'right' in cop_data:
            left_cop_x = cop_data['left']['cop_x']
            left_cop_y = cop_data['left']['cop_y']
            right_cop_x = cop_data['right']['cop_x']
            right_cop_y = cop_data['right']['cop_y']
            
            # Usuń NaN
            left_valid = ~(np.isnan(left_cop_x) | np.isnan(left_cop_y))
            right_valid = ~(np.isnan(right_cop_x) | np.isnan(right_cop_y))
            
            if np.sum(left_valid) > 0:
                ax6.plot(left_cop_x[left_valid], left_cop_y[left_valid], 
                        color=self.colors['left'], label='Lewa', alpha=0.7)
            
            if np.sum(right_valid) > 0:
                ax6.plot(right_cop_x[right_valid], right_cop_y[right_valid], 
                        color=self.colors['right'], label='Prawa', alpha=0.7)
        
        ax6.set_xlabel('Pozycja X [mm]')
        ax6.set_ylabel('Pozycja Y [mm]')
        ax6.set_title('Trajektorie centrum nacisku (COP)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axis('equal')
        
        # 7. Analiza kroków
        ax7 = fig.add_subplot(gs[2, 2:])
        left_steps, _ = self.analyzer.detect_steps('left')
        right_steps, _ = self.analyzer.detect_steps('right')
        
        ax7.plot(self.analyzer.time, left_total, label='Lewa', color=self.colors['left'], alpha=0.7)
        ax7.plot(self.analyzer.time, right_total, label='Prawa', color=self.colors['right'], alpha=0.7)
        
        # Zaznacz kroki
        for start, end in left_steps:
            ax7.axvspan(self.analyzer.time[start], self.analyzer.time[end], 
                    alpha=0.2, color=self.colors['left'])
        
        for start, end in right_steps:
            ax7.axvspan(self.analyzer.time[start], self.analyzer.time[end], 
                    alpha=0.2, color=self.colors['right'])
        
        ax7.set_xlabel('Czas [s]')
        ax7.set_ylabel('Siła całkowita')
        ax7.set_title('Wykrywanie kroków')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Korelacje między czujnikami
        ax8 = fig.add_subplot(gs[3, 0])
        correlation_results = self.analyzer.analyze_foot_correlation()
        sensors = list(correlation_results['sensor_correlations'].keys())
        correlations = [correlation_results['sensor_correlations'][s]['correlation'] for s in sensors]
        
        bars = ax8.bar(sensors, correlations, color=self.colors['both'])
        ax8.set_title('Korelacje czujników L-P')
        ax8.set_ylabel('Korelacja')
        ax8.set_xlabel('Czujnik')
        ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax8.grid(True, alpha=0.3)
        
        # 9. Wykres radarowy asymetrii
        ax9 = fig.add_subplot(gs[3, 1], projection='polar')
        
        # Przygotuj dane dla wykresu radarowego
        categories = []
        values = []
        
        if 'force_symmetry' in symmetry_results:
            categories.append('Siły')
            values.append(min(symmetry_results['force_symmetry']['total_force_symmetry_index'], 50))
        
        if 'temporal_symmetry' in symmetry_results and 'temporal_symmetry_index' in symmetry_results['temporal_symmetry']:
            categories.append('Czasowa')
            values.append(min(symmetry_results['temporal_symmetry']['temporal_symmetry_index'], 50))
        
        if 'spatial_symmetry' in symmetry_results:
            categories.append('Przestrzenna')
            values.append(min(symmetry_results['spatial_symmetry']['mean_spatial_asymmetry'], 50))
        
        if categories and values:
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Zamknij wykres
            angles += angles[:1]
            
            ax9.plot(angles, values, color=self.colors['both'], linewidth=2, marker='o')
            ax9.fill(angles, values, color=self.colors['both'], alpha=0.25)
            ax9.set_xticks(angles[:-1])
            ax9.set_xticklabels(categories)
            ax9.set_ylim(0, 50)
            ax9.set_title('Profil asymetrii', pad=20)
        
        # 10. Statystyki i podsumowanie
        ax10 = fig.add_subplot(gs[3, 2:])
        ax10.axis('off')
        
        # Przygotuj tekst podsumowania
        summary_text = []
        summary_text.append("📊 PODSUMOWANIE ANALIZY")
        summary_text.append("=" * 25)
        summary_text.append("")
        
        # Podstawowe statystyki
        summary_text.append(f"⏱️ Czas pomiaru: {self.analyzer.time[-1] - self.analyzer.time[0]:.1f} s")
        summary_text.append(f"📈 Częstotliwość: {self.analyzer.sampling_rate:.1f} Hz")
        summary_text.append("")
        
        # Kroki
        summary_text.append(f"👣 Kroki:")
        summary_text.append(f"   Lewa: {len(left_steps)}")
        summary_text.append(f"   Prawa: {len(right_steps)}")
        summary_text.append("")
        
        # Korelacja
        total_corr = correlation_results['total_force_correlation']['correlation']
        summary_text.append(f"🔗 Korelacja stóp: {total_corr:.3f}")
        
        # Symetria
        if 'overall_asymmetry_index' in symmetry_results:
            asymmetry = symmetry_results['overall_asymmetry_index']
            summary_text.append(f"⚖️ Asymetria: {asymmetry:.1f}%")
            
            # Ocena
            summary_text.append("")
            if asymmetry < 5:
                summary_text.append("✅ Bardzo dobra symetria")
            elif asymmetry < 10:
                summary_text.append("✅ Dobra symetria")
            elif asymmetry < 20:
                summary_text.append("⚠️ Umiarkowana asymetria")
            else:
                summary_text.append("❌ Znacząca asymetria")
        
        # COP
        if 'left' in cop_data and 'right' in cop_data:
            summary_text.append("")
            summary_text.append(f"🎯 COP - Długość ścieżki:")
            summary_text.append(f"   Lewa: {cop_data['left']['path_length']:.0f} mm")
            summary_text.append(f"   Prawa: {cop_data['right']['path_length']:.0f} mm")
        
        ax10.text(0.05, 0.95, '\n'.join(summary_text), transform=ax10.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background'], alpha=0.8))
        
        # Tytuł główny
        plt.suptitle(f'Kompleksowy Dashboard Analizy Biomechanicznej - {self.analyzer.sheet_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()