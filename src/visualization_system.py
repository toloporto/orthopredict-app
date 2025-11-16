# orthopredict_app/src/visualization_system.py
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, List, Optional, Tuple
import tempfile
import os

class OrthoVisualizationSystem:
    """Sistema modernizado de visualización con Plotly para OrthoPredict"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#F44336',
            'info': '#2196F3',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
        
    def crear_grafico_prediccion(self, prediccion_data: Dict, paciente_data: Dict) -> go.Figure:
        """Crear gráfico interactivo de predicción de duración"""
        prediccion = prediccion_data.get('prediccion', 0)
        min_val = prediccion_data.get('intervalo_min', 0)
        max_val = prediccion_data.get('intervalo_max', 0)
        confianza = prediccion_data.get('confianza', 75)
        
        # Datos para el gráfico
        categorias = ['Mínimo', 'Predicción', 'Máximo']
        valores = [min_val, prediccion, max_val]
        colores = [self.color_palette['warning'], self.color_palette['success'], self.color_palette['danger']]
        
        fig = go.Figure()
        
        # Barras principales
        fig.add_trace(go.Bar(
            x=categorias,
            y=valores,
            marker_color=colores,
            text=[f'{v} meses' for v in valores],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Duración: %{y} meses<extra></extra>'
        ))
        
        # Línea de referencia para la predicción
        fig.add_hline(
            y=prediccion, 
            line_dash="dash", 
            line_color=self.color_palette['primary'],
            annotation_text=f"Predicción: {prediccion} meses",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=dict(
                text=f'📈 Predicción de Duración del Tratamiento<br><sub>Confianza del modelo: {confianza}%</sub>',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Escenarios",
            yaxis_title="Duración (meses)",
            showlegend=False,
            template="plotly_white",
            height=400,
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        return fig
    
    def crear_analisis_complejidad(self, paciente_data: Dict) -> go.Figure:
        """Crear análisis visual de complejidad del caso"""
        parametros = {
            'Apiñamiento': paciente_data.get('apiñamiento_mm', 0),
            'Sobremordida': paciente_data.get('sobremordida_mm', 0),
            'Sobresalte': paciente_data.get('sobresalte_mm', 0)
        }
        
        # Calcular scores de complejidad (0-10)
        complejidad_apiñamiento = min(10, (parametros['Apiñamiento'] - 4) * 2.5)
        complejidad_sobremordida = min(10, abs(parametros['Sobremordida'] - 2.5) * 2)
        complejidad_sobresalte = min(10, abs(parametros['Sobresalte'] - 3.0) * 1.5)
        
        scores = [complejidad_apiñamiento, complejidad_sobremordida, complejidad_sobresalte]
        parametros_nombres = list(parametros.keys())
        valores_reales = list(parametros.values())
        
        fig = go.Figure()
        
        # Gráfico de radar para complejidad
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],  # Cerrar el radar
            theta=parametros_nombres + [parametros_nombres[0]],
            fill='toself',
            fillcolor='rgba(46, 134, 171, 0.3)',
            line=dict(color=self.color_palette['primary'], width=2),
            name='Complejidad'
        ))
        
        # Áreas de referencia
        fig.add_trace(go.Scatterpolar(
            r=[3, 3, 3, 3],
            theta=parametros_nombres + [parametros_nombres[0]],
            fill='toself',
            fillcolor='rgba(76, 175, 80, 0.1)',
            line=dict(color='green', width=1, dash='dot'),
            name='Baja Complejidad'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[7, 7, 7, 7],
            theta=parametros_nombres + [parametros_nombres[0]],
            fill='toself',
            fillcolor='rgba(255, 152, 0, 0.1)',
            line=dict(color='orange', width=1, dash='dot'),
            name='Alta Complejidad'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickvals=[0, 3, 7, 10],
                    ticktext=['0', 'Baja', 'Media', 'Alta']
                )
            ),
            title=dict(
                text='🎯 Análisis de Complejidad del Caso',
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=500
        )
        
        return fig
    
    def crear_comparativa_casos(self, paciente_data: Dict, datos_comparativos: pd.DataFrame = None) -> go.Figure:
        """Crear gráfico comparativo con casos similares"""
        if datos_comparativos is None:
            # Datos de ejemplo para demostración
            datos_comparativos = self._generar_datos_comparativos()
        
        apiñamiento_paciente = paciente_data.get('apiñamiento_mm', 6)
        duracion_predicha = 18  # Valor por defecto
        
        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribución por Apiñamiento', 'Duración vs Apiñamiento'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Histograma de distribución
        fig.add_trace(
            go.Histogram(
                x=datos_comparativos['apiñamiento_mm'],
                nbinsx=10,
                marker_color=self.color_palette['primary'],
                opacity=0.7,
                name='Casos Similares'
            ),
            row=1, col=1
        )
        
        # Añadir línea para el paciente actual
        fig.add_vline(
            x=apiñamiento_paciente, 
            line_dash="dash", 
            line_color=self.color_palette['danger'],
            annotation_text="Tu caso",
            row=1, col=1
        )
        
        # Scatter plot de duración vs apiñamiento
        fig.add_trace(
            go.Scatter(
                x=datos_comparativos['apiñamiento_mm'],
                y=datos_comparativos['duracion_meses'],
                mode='markers',
                marker=dict(
                    color=datos_comparativos['complejidad'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Complejidad")
                ),
                hovertemplate='<b>Apiñamiento:</b> %{x} mm<br><b>Duración:</b> %{y} meses<extra></extra>',
                name='Casos Históricos'
            ),
            row=1, col=2
        )
        
        # Añadir punto para el paciente actual
        fig.add_trace(
            go.Scatter(
                x=[apiñamiento_paciente],
                y=[duracion_predicha],
                mode='markers',
                marker=dict(
                    size=15,
                    color=self.color_palette['danger'],
                    line=dict(width=2, color='white')
                ),
                name='Tu Caso',
                hovertemplate='<b>Tu Caso</b><br>Apiñamiento: %{x} mm<br>Duración Predicha: %{y} meses<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="📊 Comparativa con Casos Similares",
            height=400,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Apiñamiento (mm)", row=1, col=1)
        fig.update_xaxes(title_text="Apiñamiento (mm)", row=1, col=2)
        fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
        fig.update_yaxes(title_text="Duración (meses)", row=1, col=2)
        
        return fig
    
    def crear_evolucion_performance(self, metricas_historicas: List[Dict]) -> go.Figure:
        """Crear gráfico de evolución del performance del modelo"""
        if not metricas_historicas:
            return self._crear_grafico_vacio("No hay datos de performance históricos")
        
        fechas = []
        r2_scores = []
        mae_scores = []
        
        for metrica in metricas_historicas:
            if 'metricas_performance' in metrica and 'r2' in metrica['metricas_performance']:
                fechas.append(metrica['fecha'][:10])  # Solo la fecha
                r2_scores.append(metrica['metricas_performance']['r2'])
                mae_scores.append(metrica['metricas_performance'].get('mae', 0))
        
        if not fechas:
            return self._crear_grafico_vacio("No hay métricas de performance disponibles")
        
        fig = sp.make_subplots(
            rows=2, cols=1,
            subplot_titles=('Evolución del R²', 'Evolución del MAE (meses)'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Gráfico de R²
        fig.add_trace(
            go.Scatter(
                x=fechas,
                y=r2_scores,
                mode='lines+markers',
                line=dict(color=self.color_palette['success'], width=3),
                marker=dict(size=8),
                name='R² Score',
                hovertemplate='<b>Fecha:</b> %{x}<br><b>R²:</b> %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Línea de umbral para R²
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="Umbral Mínimo", row=1, col=1)
        
        # Gráfico de MAE
        fig.add_trace(
            go.Scatter(
                x=fechas,
                y=mae_scores,
                mode='lines+markers',
                line=dict(color=self.color_palette['warning'], width=3),
                marker=dict(size=8),
                name='MAE',
                hovertemplate='<b>Fecha:</b> %{x}<br><b>MAE:</b> %{y:.2f} meses<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title_text="📈 Evolución del Performance del Modelo ML",
            height=600,
            showlegend=False,
            template="plotly_white"
        )
        
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        
        return fig
    
    def exportar_grafico_a_imagen(self, fig: go.Figure, formato: str = 'png', 
                                ancho: int = 800, alto: int = 600) -> bytes:
        """Exportar gráfico Plotly a imagen para PDFs"""
        try:
            # Usar kaleido para exportación
            img_bytes = fig.to_image(format=formato, width=ancho, height=alto, scale=2)
            return img_bytes
        except Exception as e:
            print(f"Error exportando gráfico: {e}")
            # Fallback: crear gráfico simple
            return self._crear_grafico_fallback().to_image(format=formato, width=ancho, height=alto)
    
    def crear_grafico_para_pdf(self, fig: go.Figure) -> str:
        """Crear imagen base64 para incrustar en PDFs de ReportLab"""
        try:
            img_bytes = self.exportar_grafico_a_imagen(fig, 'png', 600, 400)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_base64
        except Exception as e:
            print(f"Error creando gráfico para PDF: {e}")
            return ""
    
    def _generar_datos_comparativos(self) -> pd.DataFrame:
        """Generar datos comparativos de ejemplo"""
        np.random.seed(42)
        n_muestras = 100
        
        datos = {
            'apiñamiento_mm': np.random.normal(6, 1, n_muestras),
            'duracion_meses': np.random.normal(18, 3, n_muestras),
            'complejidad': np.random.uniform(0, 10, n_muestras)
        }
        
        # Ajustar rangos
        datos['apiñamiento_mm'] = np.clip(datos['apiñamiento_mm'], 4, 8)
        datos['duracion_meses'] = np.clip(datos['duracion_meses'], 12, 30)
        
        return pd.DataFrame(datos)
    
    def _crear_grafico_vacio(self, mensaje: str) -> go.Figure:
        """Crear gráfico vacío con mensaje informativo"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=mensaje,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=300
        )
        
        return fig
    
    def _crear_grafico_fallback(self) -> go.Figure:
        """Crear gráfico de fallback simple"""
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3]))
        fig.update_layout(title="Gráfico de Ejemplo")
        return fig

    def crear_dashboard_analitico(self, datos_pacientes: pd.DataFrame) -> go.Figure:
        """Dashboard avanzado de analytics con subplots."""
        if datos_pacientes.empty or len(datos_pacientes) < 5:
            return self._crear_grafico_vacio("Datos insuficientes para el dashboard analítico (se requieren al menos 5 pacientes).")

        # Asegurar que las columnas de fecha son datetime y complejidad es numérica
        if 'fecha_creacion' in datos_pacientes.columns:
            datos_pacientes['fecha_creacion'] = pd.to_datetime(datos_pacientes['fecha_creacion'], errors='coerce')
        if 'complejidad_score' not in datos_pacientes.columns:
            datos_pacientes['complejidad_score'] = 0
        datos_pacientes['complejidad_score'] = pd.to_numeric(datos_pacientes['complejidad_score'], errors='coerce').fillna(0)

        # Crear figura con subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "1. Tendencia de Nuevos Pacientes",
                "2. Duración Predicha por Complejidad",
                "3. Distribución de Scores de Complejidad",
                "4. Edad vs. Complejidad"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # 1. Análisis de tendencias temporales
        if 'fecha_creacion' in datos_pacientes.columns:
            tendencia_temporal = datos_pacientes.set_index('fecha_creacion').resample('M').size()
            fig.add_trace(go.Scatter(
                x=tendencia_temporal.index, y=tendencia_temporal.values,
                mode='lines+markers', name='Nuevos Pacientes',
                line=dict(color=self.color_palette['primary'])
            ), row=1, col=1)

        # 2. Segmentación por complejidad
        datos_pacientes['grupo_complejidad'] = pd.cut(
            datos_pacientes['complejidad_score'],
            bins=[0, 40, 70, 101],
            labels=['Baja', 'Media', 'Alta'],
            right=False
        )
        fig.add_trace(go.Box(
            x=datos_pacientes['grupo_complejidad'],
            y=datos_pacientes['duracion_predicha'],
            marker_color=self.color_palette['secondary'],
            name='Duración por Complejidad'
        ), row=1, col=2)

        # 3. Distribución de complejidad
        fig.add_trace(go.Histogram(
            x=datos_pacientes['complejidad_score'],
            marker_color=self.color_palette['info'],
            name='Distribución Complejidad'
        ), row=2, col=1)

        # 4. Análisis de performance del modelo (Edad vs Complejidad)
        fig.add_trace(go.Scatter(
            x=datos_pacientes['edad'],
            y=datos_pacientes['complejidad_score'],
            mode='markers',
            marker=dict(color=datos_pacientes['duracion_predicha'], colorscale='Viridis', showscale=True, colorbar=dict(title='Duración Predicha')),
            name='Edad vs Complejidad'
        ), row=2, col=2)

        # Actualizar layout general
        fig.update_layout(
            title_text='🚀 Dashboard Analítico de Pacientes',
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        return fig

# Instancia global del sistema de visualización
viz_system = OrthoVisualizationSystem()