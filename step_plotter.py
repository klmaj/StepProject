from step_analyzer import FootSensorAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64

class StepPlotter:
    def __init__(self, time, analyzer:FootSensorAnalyzer):
        self.analyzer = analyzer
        self.time = time
        self.left = analyzer.left
        self.right = analyzer.right

    def encode_image(self, img_path):
            with open(img_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            return "data:image/png;base64," + encoded
        
    def plot_signals(self, czas_max=None, return_fig=False):
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
            title = "Przebiegi wartości siły nacisku od czasu"

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

        # Zakoduj obrazki stóp
        left_img = self.encode_image("left_foot.png")
        right_img = self.encode_image("right_foot.png")

        # Dodaj obrazy jako layout_image
        fig.add_layout_image(
            dict(
                source=left_img,
                xref="paper", yref="paper",
                x=1.12, y=0.9,  # obok górnego wykresu
                sizex=0.12, sizey=0.3,
                xanchor="right", yanchor="top",
                layer="above"
            )
        )

        fig.add_layout_image(
            dict(
                source=right_img,
                xref="paper", yref="paper",
                x=1.12, y=0.35,  # obok dolnego wykresu
                sizex=0.12, sizey=0.3,
                xanchor="right", yanchor="top",
                layer="above"
            )
        )


        # Interaktywność i styl
        fig.update_layout(
            title=title,
            hovermode="x unified",
            template="plotly_white",
            dragmode='zoom',
            margin=dict(t=50, b=50, r=200),    # zostaw miejsce na obrazki na dole
            #width=1400,
            height=900,
        )

        fig.update_yaxes(title_text="Lewa stopa F [N]", fixedrange=False, row=1, col=1)
        fig.update_yaxes(title_text="Prawa stopa F [N]", fixedrange=False, row=2, col=1)
        fig.update_xaxes(title_text="Czas [s]", row=2, col=1)

        if return_fig:
            return fig
        else:
            fig.show()
    
    def plot_step_detection_plotly(self, foot='left', return_fig=False):
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
        
        if return_fig:
            return fig
        else:
            fig.show()

        return steps
    
    def plot_pressure_heatmap_plotly(self, foot='left', return_fig=False):
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
        
        if return_fig:
            return fig
        else:
            fig.show()
    
    def plot_step_detection(self, foot='left', figsize=(12, 8), return_fig=False):
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
        if return_fig:
            return plt
        else:
            plt.show()
        
        print(f"Wykryto {len(steps)} kroków dla {foot} stopy")
        return steps
    
    def plot_pressure_heatmap_both_feet(self, return_fig=False):
        """Interaktywna mapa ciepła pokazująca rozkład nacisku dla obu stóp (8x2)"""
    
        # Utwórz macierz 8x2: 8 pozycji czujników x 2 stopy (lewa, prawa)
        heatmap_matrix = np.zeros((8, 2))
    
        # Kolejność czujników: Heel Med, Heel Lat, Mid Med, Mid Lat, Fore Med, Fore Lat, Toe 1, Toe 2-5
        sensor_labels = [
            1, 2,  # Heel Med, Heel Lat
            3, 4,  # Mid Med, Mid Lat
            5, 6,  # Fore Med, Fore Lat
            7, 8   # Toe 1, Toe 2-5
        ]
    
        # Wypełnij macierz danymi z lewej i prawej stopy
        
        left_means = self.analyzer.mean_max_pressure_per_step('left') # np.mean(self.left[:, sensor_idx])   # lewa stopa
        right_means = self.analyzer.mean_max_pressure_per_step('right') #np.mean(self.right[:, sensor_idx])  # prawa stopa

        heatmap_matrix = np.column_stack((left_means, right_means))


        left_img_encoded = self.encode_image("left_foot.png")
        right_img_encoded = self.encode_image("right_foot.png")

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_matrix,
            x=["Lewa stopa", "Prawa stopa"],
            y=sensor_labels,
            colorscale='YlOrRd',
            text=np.round(heatmap_matrix, 1),
            texttemplate="%{text:.1f}",
            textfont={"size": 12},
            hovertemplate='%{y}<br>%{x}<br>Średnia siła: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title="Średni rozkład nacisku na stopie",
            xaxis_title=" ",
            yaxis_title="Czujnik",
            yaxis_autorange='reversed',  # żeby zachować podobny układ do matplotlib
            margin=dict(t=50, b=250),    # zostaw miejsce na obrazki na dole
            width=800,
            height=900,
            template="plotly_white",
            images=[
                dict(
                    source=left_img_encoded,
                    xref="paper", yref="paper",
                    x=0.22, y=-0.4,  # pozycja w osi papieru (x od lewej do prawej 0-1, y od dołu do góry)
                    sizex=0.3, sizey=0.35,
                    xanchor="center", yanchor="bottom",
                    layer="above"
                ),
                dict(
                    source=right_img_encoded,
                    xref="paper", yref="paper",
                    x=0.78, y=-0.4,
                    sizex=0.3, sizey=0.35,
                    xanchor="center", yanchor="bottom",
                    layer="above"
                )
            ]
        )

        if return_fig:
            return fig
        else:
            fig.show()

   
    def plot_step_periods(self, return_fig=False):
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

        if return_fig:
            return fig
        else:
            fig.show()

    def plot_cross_correlation_matrix(self, return_fig=False):
        """
        Rysuje mapę korelacji pomiędzy czujnikami lewej i prawej stopy (8x8),
        z rysunkami stóp po prawej stronie.
        """
        corr_matrix = self.analyzer.compute_cross_correlation_matrix()
        sensor_labels = [f"Czujnik {i+1}" for i in range(8)]

        # Zakodowane obrazki
        left_img_encoded = self.encode_image("left_foot.png")
        right_img_encoded = self.encode_image("right_foot.png")

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=[f"P{i+1}" for i in range(8)],
            y=[f"L{i+1}" for i in range(8)],
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title='Korelacja'),
            text=np.round(corr_matrix, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 12},
            hovertemplate='Lewa %{x}<br>Prawa %{y}<br>Korelacja: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title="Korelacja pomiędzy czujnikami lewej i prawej stopy",
            yaxis_title="Czujniki lewej stopy",
            xaxis_title="Czujniki prawej stopy",
            template="plotly_white",
            width=1000,
            height=700,
            margin=dict(t=50, b=50, r=200),  # zostaw miejsce po prawej
            images=[
                dict(
                    source=left_img_encoded,
                    xref="paper", yref="paper",
                    x=1.26, y=0.9,
                    sizex=0.12, sizey=0.3,
                    xanchor="right", yanchor="top",
                    layer="above"
                ),
                dict(
                    source=right_img_encoded,
                    xref="paper", yref="paper",
                    x=1.26, y=0.35,
                    sizex=0.12, sizey=0.3,
                    xanchor="right", yanchor="top",
                    layer="above"
                )
            ]
        )

        if return_fig:
            return fig
        else:
            fig.show()


    def plot_heatmap_and_correlation_side_by_side(self, return_fig=False):
        """Łączy mapę nacisku i korelację czujników w jednej karcie, z obrazkami stóp"""

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np

        # --- Dane ---
        left_means = self.analyzer.mean_max_pressure_per_step('left')
        right_means = self.analyzer.mean_max_pressure_per_step('right')
        heatmap_matrix = np.column_stack((left_means, right_means))
        corr_matrix = self.analyzer.compute_cross_correlation_matrix()

        # --- Zakodowane obrazki ---
        left_img_encoded = self.encode_image("left_foot.png")
        right_img_encoded = self.encode_image("right_foot.png")

        # --- Subplots ---
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Średni rozkład nacisku", "Korelacja czujników L vs P"),
            column_widths=[0.5, 0.5]
        )

        # --- Heatmapa nacisku ---
        fig.add_trace(go.Heatmap(
            z=heatmap_matrix,
            x=["Lewa stopa", "Prawa stopa"],
            y=[f"Czujnik {i+1}" for i in range(8)],
            colorscale='YlOrRd',
            zmin=np.min(heatmap_matrix), zmax=np.max(heatmap_matrix),
            showscale=False,
            text=np.round(heatmap_matrix, 1),
            texttemplate="%{text:.1f}",
            hovertemplate='%{y}<br>%{x}<br>Średnia siła: %{z:.2f}<extra></extra>'
        ), row=1, col=1)

        # --- Heatmapa korelacji ---
        fig.add_trace(go.Heatmap(
            z=corr_matrix,
            x=[f"L{i+1}" for i in range(8)],
            y=[f"P{i+1}" for i in range(8)],
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title='Korelacja'),
            text=np.round(corr_matrix, 2),
            texttemplate="%{text:.2f}",
            hovertemplate='Lewa %{x}<br>Prawa %{y}<br>Korelacja: %{z:.2f}<extra></extra>'
        ), row=1, col=2)

        # --- Layout + obrazki ---
        fig.update_layout(
            title="Porównanie: nacisk vs korelacja między czujnikami",
            height=900,
            width=1300,
            template="plotly_white",
            margin=dict(t=50, b=250),
            images=[
                dict(
                    source=left_img_encoded,
                    xref="paper", yref="paper",
                    x=0.22, y=-0.35,
                    sizex=0.3, sizey=0.35,
                    xanchor="center", yanchor="bottom",
                    layer="above"
                ),
                dict(
                    source=right_img_encoded,
                    xref="paper", yref="paper",
                    x=0.78, y=-0.35,
                    sizex=0.3, sizey=0.35,
                    xanchor="center", yanchor="bottom",
                    layer="above"
                )
            ]
        )

        if return_fig:
            return fig
        else:
            fig.show()

    def plot_symmetry_indices(self, return_fig=False):
        """
        Rysuje globalny i czujnikowy wskaźnik symetrii.
        - słupek dla każdego czujnika
        - osobny pasek dla globalnej symetrii
        """
        sym_data = self.analyzer.compute_symmetry_indices()
        sensor_sym = sym_data['sensor_symmetry']
        global_sym = sym_data['global_symmetry']

        labels = [f"Czujnik {k}" for k in sensor_sym.keys()]
        values = list(sensor_sym.values())

        fig = go.Figure()

        # Wykres słupkowy dla czujników
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            name="Symetria czujnikowa [%]",
            marker_color='steelblue'
        ))

        # Linia pozioma z globalnym wskaźnikiem
        fig.add_trace(go.Scatter(
            x=labels,
            y=[global_sym] * len(labels),
            name=f"Symetria globalna ({global_sym:.1f}%)",
            mode='lines',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title="Wskaźniki symetrii siły nacisku",
            yaxis_title="Symetria [%]",
            xaxis_title="Czujniki",
            template="plotly_white",
            hovermode="x unified",
            height=500
        )

        if return_fig:
            return fig
        else:
            fig.show()


    def plot_correlation_heatmaps(self, mean_corrs, max_corrs):
        """
        Rysuje dwie mapy ciepła (heatmapy) korelacji między lewą a prawą stopą:
        - jedna dla średniego nacisku
        - druga dla maksymalnego nacisku

        Params:
            mean_corrs: dict z korelacjami (np. {'czujnik_1': 0.82, ...}) dla średniego sygnału
            max_corrs: dict jak wyżej, ale dla maksymalnego sygnału
        """
        sensors = list(mean_corrs.keys())
        mean_vals = [mean_corrs[s] for s in sensors]
        max_vals = [max_corrs[s] for s in sensors]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        sns.heatmap(
            np.array(mean_vals).reshape(1, -1), 
            annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=sensors, yticklabels=["mean"], 
            ax=axes[0], vmin=-1, vmax=1, cbar=True
        )
        axes[0].set_title("Korelacja L vs P (średni nacisk)")

        sns.heatmap(
            np.array(max_vals).reshape(1, -1), 
            annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=sensors, yticklabels=["max"], 
            ax=axes[1], vmin=-1, vmax=1, cbar=True
        )
        axes[1].set_title("Korelacja L vs P (maksymalny nacisk)")

        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


    def plot_full_corr_heatmap_plotly(self, corr_mean, corr_max, step_range=(0, 0)):
        """
        Rysuje dwie heatmapy (obok siebie) pokazujące korelację czujników lewej i prawej stopy:
        - po lewej: korelacja na podstawie średniego nacisku
        - po prawej: korelacja na podstawie maksymalnego nacisku
        """
        sensors_L = [f"L{i+1}" for i in range(8)]
        sensors_P = [f"P{i+1}" for i in range(8)]
        step_info = f"Kroki {step_range[0]}–{step_range[1]}"

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Korelacja L vs P (średni nacisk)<br><sup>{step_info}</sup>",
                f"Korelacja L vs P (maksymalny nacisk)<br><sup>{step_info}</sup>"
            ),  
            horizontal_spacing=0.15
        )

        # Heatmapa dla średnich wartości
        fig.add_trace(
            go.Heatmap(
                z=corr_mean,
                x=sensors_P,
                y=sensors_L,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                colorbar=dict(title="Korelacja", x=0.47),
                text=np.round(corr_mean, 2),
                texttemplate="%{text:.2f}"
            ),
            row=1, col=1
        )

        # Heatmapa dla maksymalnych wartości
        fig.add_trace(
            go.Heatmap(
                z=corr_max,
                x=sensors_P,
                y=sensors_L,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                showscale=False,
                text=np.round(corr_max, 2),
                texttemplate="%{text:.2f}"
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=600,
            width=1500,
            margin=dict(t=100, b=50, r=200),  # zostaw miejsce po prawej
            title_text="Macierz korelacji czujników L vs P (średnia i maksymalna wartość w krokach)",
            template="plotly_white"
        )

        fig.show()


    def plot_si_map_heatmaps(self, si_matrix, mapd_matrix):
        """
        Rysuje dwie heatmapy 8x8: Symmetry Index (SI) i MAPD.
        Wiersze: czujniki lewej stopy (L1–L8), kolumny: czujniki prawej stopy (P1–P8).
        """
        sensors_x = [f"P{j+1}" for j in range(8)]
        sensors_y = [f"L{i+1}" for i in range(8)]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Symmetry Index (SI)", "MAPD"),
            horizontal_spacing=0.15
        )

        # SI heatmap
        fig.add_trace(go.Heatmap(
            z=si_matrix,
            x=sensors_x,
            y=sensors_y,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="SI (%)", x=0.45),
            text=np.round(si_matrix, 2),
            texttemplate="%{text:.2f}"
        ), row=1, col=1)

        # MAPD heatmap
        fig.add_trace(go.Heatmap(
            z=mapd_matrix,
            x=sensors_x,
            y=sensors_y,
            colorscale="YlOrRd",
            colorbar=dict(title="MAPD (%)", x=1.0),
            text=np.round(mapd_matrix, 2),
            texttemplate="%{text:.2f}"
        ), row=1, col=2)

        fig.update_layout(
            title="Mapa Symmetry Index i MAPD między czujnikami L i P",
            height=800,
            width=1600,
            margin=dict(t=100, b=250, r=200),  # zostaw miejsce po prawej
            template="plotly_white"
        )

       # Adnotacja pod wykresem SI
        fig.add_annotation(
            text=(
                "<b>SI – Symmetry Index</b><br>"
                "Wzór: SI = 100 × (L - P) / (0.5 × (L + P))<br>"
                "Interpretacja: <br>"
                "SI bliskie 0% oznacza symetrię. <br>"
                "SI > 0: przewaga lewej stopy, "
                "SI < 0: przewaga prawej stopy."
            ),
            showarrow=False,
            xref="paper", yref="paper",
             x=0.05, y=-0.25,  # Pozycja pod pierwszym wykresem
            align="left",
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=0
        )

        # Adnotacja pod wykresem MAPD
        fig.add_annotation(
            text=(
                "<b>MAPD – Mean Absolute Percentage Difference</b><br>"
                "Wzór: MAPD = 100 × |L - P| / ((L + P) / 2)<br>"
                "Interpretacja:<br>"
                "MAPD mierzy względną różnicę między czujnikami <br>"
                "bez względu na kierunek – im wyższa wartość, tym większa asymetria."
            ),
            showarrow=False,
            xref="paper", yref="paper",
            x=0.875, y=-0.25,  # Pozycja pod drugim wykresem
            align="left",
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=0
        )


        fig.show()

    def plot_global_signals(self, global_left, global_right):
        """
        Rysuje globalny sygnał nacisku lewej i prawej stopy w czasie.
        """

        avg_left = np.mean(global_left)
        avg_right = np.mean(global_right)
        sample_count = max(len(global_left), len(global_right))

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=global_left, mode='lines', name='Lewa stopa', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=global_right, mode='lines', name='Prawa stopa', line=dict(color='red')))

        # Średnia lewa
        fig.add_trace(go.Scatter(
            y=[avg_left] * sample_count, mode='lines',
            name='Średnia lewa', line=dict(color='blue', dash='dash')
        ))

        # Średnia prawa
        fig.add_trace(go.Scatter(
            y=[avg_right] * sample_count, mode='lines',
            name='Średnia prawa', line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title="Globalna siła nacisku lewej i prawej stopy w czasie",
            xaxis_title="Czas (próbki)",
            yaxis_title="Suma nacisków z 8 czujników",
            template="plotly_white"
        )
        fig.show()

    