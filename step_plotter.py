from step_analyzer import FootSensorAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class StepPlotter:
    def __init__(self, time, analyzer:FootSensorAnalyzer):
        self.analyzer = analyzer
        self.time = time
        self.left = analyzer.left
        self.right = analyzer.right
        
    def plot_signals(self, czas_max=None):
        """Interaktywny wykres sygnałów z czujników używając Plotly"""
        # Filtrowanie danych wg zakresu czasu
        if czas_max is not None:
            mask = self.time <= czas_max
            time_plot = self.time[mask]
            left_plot = self.left[mask]
            right_plot = self.right[mask]
            title = f"Dane do {czas_max} s"
        else:
            time_plot = self.time
            left_plot = self.left
            right_plot = self.right
            title = "Pełne dane"

        # Kolory
        colors = ['red', 'yellow', 'blue', 'darkgreen', 'brown', 'cyan', 'magenta', 'lightgreen']

        # Wykres
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=["Lewa stopa", "Prawa stopa"]
        )

        # Lewa stopa
        for i in range(left_plot.shape[1]):
            fig.add_trace(
                go.Scatter(x=time_plot, y=left_plot[:, i], mode='lines',
                           name=f'L{i+1}', line=dict(color=colors[i])),
                row=1, col=1
            )

        # Prawa stopa
        for i in range(right_plot.shape[1]):
            fig.add_trace(
                go.Scatter(x=time_plot, y=right_plot[:, i], mode='lines',
                           name=f'P{i+1}', line=dict(color=colors[i])),
                row=2, col=1
            )

        # Suwak czasu na dolnej osi
        fig.update_xaxes(
            rangeslider=dict(visible=True),
            row=2, col=1
        )

        # Interaktywność i styl
        fig.update_layout(
            height=900,
            title=title,
            hovermode="x unified",
            template="plotly_white",
            dragmode='zoom'
        )

        fig.update_yaxes(title_text="Lewa stopa F [N]", fixedrange=False, row=1, col=1)
        fig.update_yaxes(title_text="Prawa stopa F [N]", fixedrange=False, row=2, col=1)
        fig.update_xaxes(title_text="Czas [s]", row=2, col=1)

        fig.show()
    
    def plot_step_detection_plotly(self, foot='left'):
        """Interaktywny wykres wykrywania kroków"""
        steps, total_force = self.analyzer.detect_steps(foot)
        
        fig = go.Figure()
        
        # Główny sygnał
        fig.add_trace(go.Scatter(
            x=self.time, 
            y=total_force, 
            mode='lines',
            name=f'Całkowita siła - {foot} stopa',
            line=dict(color='blue', width=2)
        ))
        
        # Oznacz wykryte kroki
        for i, (start, end) in enumerate(steps):
            fig.add_vrect(
                x0=self.time[start], x1=self.time[end],
                fillcolor="red", opacity=0.3,
                annotation_text=f"Krok {i+1}" if i < 5 else "",
                annotation_position="top left"
            )
        
        fig.update_layout(
            title=f'Wykrywanie kroków - {foot} stopa (wykryto {len(steps)} kroków)',
            xaxis_title='Czas [s]',
            yaxis_title='Całkowita siła',
            template="plotly_white",
            hovermode="x"
        )
        
        fig.show()
        return steps
    
    def plot_pressure_heatmap_plotly(self, foot='left'):
        """Interaktywna mapa ciepła rozkładu nacisku"""
        if foot == 'left':
            data = self.left
        else:
            data = self.right
        
        # Utwórz macierz reprezentującą stopę
        foot_matrix = np.zeros((4, 2))
        sensor_positions = [
            (3, 0), (3, 1),  # Heel Med, Heel Lat
            (2, 0), (2, 1),  # Mid Med, Mid Lat  
            (1, 0), (1, 1),  # Fore Med, Fore Lat
            (0, 0), (0, 1)   # Toe 1, Toe 2-5
        ]
        
        # Wypełnij macierz średnimi wartościami
        for i, (row, col) in enumerate(sensor_positions):
            foot_matrix[row, col] = np.mean(data[:, i])
        
        fig = go.Figure(data=go.Heatmap(
            z=foot_matrix,
            x=['Wewnętrzna', 'Zewnętrzna'],
            y=['Palce', 'Przód', 'Środek', 'Pięta'],
            colorscale='YlOrRd',
            text=foot_matrix,
            texttemplate="%{text:.1f}",
            textfont={"size": 12},
            hovertemplate='%{y}<br>%{x}<br>Średnia siła: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Rozkład średniego nacisku - {foot} stopa',
            xaxis_title='Strona stopy',
            yaxis_title='Obszar stopy',
            template="plotly_white"
        )
        
        fig.show()
    
    def plot_step_detection(self, foot='left', figsize=(12, 8)):
        """Wykres wykrywania kroków"""
        steps, total_force = self.detect_steps(foot)
        
        plt.figure(figsize=figsize)
        plt.plot(self.time, total_force, 'b-', linewidth=2, label=f'Całkowita siła - {foot} stopa')
        
        # Oznacz wykryte kroki
        for i, (start, end) in enumerate(steps):
            plt.axvspan(self.time[start], self.time[end], alpha=0.3, color='red', 
                       label='Wykryty krok' if i == 0 else "")
        
        plt.title(f'Wykrywanie kroków - {foot} stopa')
        plt.xlabel('Czas [s]')
        plt.ylabel('Całkowita siła')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Wykryto {len(steps)} kroków dla {foot} stopy")
        return steps
    
    def plot_pressure_heatmap(self, foot='left', figsize=(10, 6)):
        """Mapa ciepła pokazująca rozkład nacisku na stopie"""
        if foot == 'left':
            data = self.left
        else:
            data = self.right
        
        # Utwórz macierz reprezentującą stopę (uproszczony kształt)
        foot_matrix = np.zeros((4, 2))  # 4 rzędy (pięta->palce), 2 kolumny (wewnętrzna/zewnętrzna)
        
        # Mapowanie czujników na pozycje w macierzy stopy
        sensor_positions = [
            (3, 0), (3, 1),  # Heel Med, Heel Lat
            (2, 0), (2, 1),  # Mid Med, Mid Lat  
            (1, 0), (1, 1),  # Fore Med, Fore Lat
            (0, 0), (0, 1)   # Toe 1, Toe 2-5
        ]
        
        # Wypełnij macierz średnimi wartościami
        for i, (row, col) in enumerate(sensor_positions):
            foot_matrix[row, col] = np.mean(data[:, i])
        
        plt.figure(figsize=figsize)
        sns.heatmap(foot_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                   xticklabels=['Wewnętrzna', 'Zewnętrzna'],
                   yticklabels=['Palce', 'Przód', 'Środek', 'Pięta'])
        plt.title(f'Rozkład średniego nacisku - {foot} stopa')
        plt.ylabel('Obszar stopy')
        plt.show()   
   
    def plot_step_periods(self):
        """
        Rysuje wykres okresu kroku (czas między kolejnymi zerami) w funkcji czasu,
        osobno dla lewej i prawej stopy, na jednym wykresie.
        """
        # Dane z analizatora
        times_left, periods_left = self.analyzer.get_step_periods("left")
        times_right, periods_right = self.analyzer.get_step_periods("right")

        # Tworzenie wykresu interaktywnego
        fig = go.Figure()

        # Lewa stopa – okresy kroków
        fig.add_trace(go.Scatter(
            x=times_left,
            y=periods_left,
            mode='markers',
            name='Lewa stopa',
            marker=dict(color='blue')
        ))

        # Prawa stopa – okresy kroków
        fig.add_trace(go.Scatter(
            x=times_right,
            y=periods_right,
            mode='markers',
            name='Prawa stopa',
            marker=dict(color='red')
        ))

        # Średnia lewa
        mean_left = np.mean(periods_left)
        fig.add_trace(go.Scatter(
            x=[times_left.min(), times_left.max()],
            y=[mean_left, mean_left],
            mode='lines',
            name=f'Średnia lewa ({mean_left:.2f}s)',
            line=dict(color='blue', dash='solid', width=1)
        ))

        # Średnia prawa
        mean_right = np.mean(periods_right)
        fig.add_trace(go.Scatter(
            x=[times_right.min(), times_right.max()],
            y=[mean_right, mean_right],
            mode='lines',
            name=f'Średnia prawa ({mean_right:.2f}s)',
            line=dict(color='red', dash='solid', width=1)
        ))

        fig.update_layout(
            title="Okresy kroków w czasie",
            xaxis_title="Czas [s]",
            yaxis_title="Okres kroku [s]",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )

        
        fig.update_yaxes(range=[0, 1.6])

        fig.show()