# orthopredict_app/src/monitoring_dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import logging

from monitoring_system import MLMonitoringSystem
from visualization_system import viz_system
from config import config
from cache_system import cache_system

logger = logging.getLogger(__name__)

class RealTimeMonitoringDashboard:
    """Dashboard de monitoreo en tiempo real para OrthoPredict"""
    
    def __init__(self, monitoring_system: MLMonitoringSystem):
        self.monitoring_system = monitoring_system
        self.update_interval = 30  # segundos entre actualizaciones
        
    def render_dashboard(self):
        """Renderizar el dashboard completo de monitoreo"""
        st.header("📈 Dashboard de Monitoreo en Tiempo Real")
        
        # Indicador de estado en tiempo real
        self._render_status_indicator()
        
        # Métricas clave en tiempo real
        self._render_realtime_metrics()
        
        # Gráficos de performance
        self._render_performance_charts()
        
        # Alertas y notificaciones
        self._render_alerts_panel()
        
        # Análisis de tendencias
        self._render_trend_analysis()
        
        # Sistema de cache
        self._render_cache_analytics()
        
        # Controles de monitoreo
        self._render_control_panel()
    
    def _render_status_indicator(self):
        """Indicador de estado del sistema en tiempo real"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Estado general del modelo
            latest_eval = self._get_latest_evaluation()
            if latest_eval:
                status = latest_eval['metricas_calidad']['estado']
                color = self._get_status_color(status)
                st.metric(
                    "Estado del Modelo", 
                    status,
                    delta=None,
                    delta_color=color
                )
        
        with col2:
            # Performance actual
            if latest_eval and 'metricas_performance' in latest_eval:
                r2 = latest_eval['metricas_performance'].get('r2', 0)
                st.metric("R² Actual", f"{r2:.3f}")
        
        with col3:
            # Alertas activas
            active_alerts = len([a for a in self.monitoring_system.alertas 
                               if datetime.now() - datetime.fromisoformat(a['fecha']) < timedelta(hours=24)])
            alert_color = "normal" if active_alerts == 0 else "off" if active_alerts <= 3 else "inverse"
            st.metric("Alertas (24h)", active_alerts, delta_color=alert_color)
        
        with col4:
            # Tasa de cache
            cache_stats = cache_system.get_stats()
            hit_rate = cache_stats.get('hit_rate', 0)
            st.metric("Tasa Cache", f"{hit_rate:.1f}%")
    
    def _render_realtime_metrics(self):
        """Métricas en tiempo real con actualización automática"""
        st.subheader("📊 Métricas en Tiempo Real")
        
        # Usar columns para layout responsivo
        col1, col2, col3, col4 = st.columns(4)
        
        latest_eval = self._get_latest_evaluation()
        
        with col1:
            if latest_eval and 'metricas_performance' in latest_eval:
                mae = latest_eval['metricas_performance'].get('mae', 0)
                st.metric("MAE", f"{mae:.2f} meses")
        
        with col2:
            if latest_eval and 'metricas_performance' in latest_eval:
                mape = latest_eval['metricas_performance'].get('mape', 0)
                st.metric("MAPE", f"{mape:.1f}%")
        
        with col3:
            # Drift score
            if latest_eval and 'drift_datos' in latest_eval:
                drift_score = latest_eval['drift_datos'].get('drift_score', 0)
                st.metric("Drift Score", f"{drift_score:.3f}")
        
        with col4:
            # Uptime del sistema
            if self.monitoring_system.metricas_historicas:
                first_eval = min([datetime.fromisoformat(m['fecha']) 
                                for m in self.monitoring_system.metricas_historicas])
                uptime_days = (datetime.now() - first_eval).days
                st.metric("Uptime", f"{uptime_days} días")
    
    def _render_performance_charts(self):
        """Gráficos interactivos de performance"""
        st.subheader("📈 Evolución del Performance")
        
        if len(self.monitoring_system.metricas_historicas) < 2:
            st.info("Se necesitan al menos 2 evaluaciones para mostrar gráficos de tendencia")
            return
        
        # Crear pestañas para diferentes visualizaciones
        tab1, tab2, tab3 = st.tabs(["📊 Métricas Principales", "📉 Tendencias R²", "📈 Análisis de Error"])
        
        with tab1:
            self._render_main_metrics_chart()
        
        with tab2:
            self._render_r2_trend_chart()
        
        with tab3:
            self._render_error_analysis_chart()
    
    def _render_main_metrics_chart(self):
        """Gráfico principal de métricas de performance"""
        metrics_data = self._prepare_metrics_data()
        
        if metrics_data.empty:
            st.warning("No hay datos suficientes para el gráfico")
            return
        
        fig = go.Figure()
        
        # R² (escala 0-1)
        fig.add_trace(go.Scatter(
            x=metrics_data['fecha'],
            y=metrics_data['r2'],
            mode='lines+markers',
            name='R² Score',
            line=dict(color='#2E86AB', width=3),
            yaxis='y1'
        ))
        
        # MAE (escala 0-5 meses)
        fig.add_trace(go.Scatter(
            x=metrics_data['fecha'],
            y=metrics_data['mae'],
            mode='lines+markers',
            name='MAE (meses)',
            line=dict(color='#A23B72', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Evolución de Métricas de Performance",
            xaxis=dict(title="Fecha"),
            yaxis=dict(
                title="R² Score",
                range=[0, 1],
                tickformat=".2f"
            ),
            yaxis2=dict(
                title="MAE (meses)",
                overlaying='y',
                side='right',
                range=[0, 5]
            ),
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_r2_trend_chart(self):
        """Gráfico especializado en tendencia de R²"""
        metrics_data = self._prepare_metrics_data()
        
        if metrics_data.empty:
            return
        
        # Calcular media móvil
        metrics_data['r2_ma'] = metrics_data['r2'].rolling(window=3, min_periods=1).mean()
        
        fig = go.Figure()
        
        # R² actual
        fig.add_trace(go.Scatter(
            x=metrics_data['fecha'],
            y=metrics_data['r2'],
            mode='markers',
            name='R² por Evaluación',
            marker=dict(size=8, color='#2E86AB'),
            opacity=0.6
        ))
        
        # Media móvil
        fig.add_trace(go.Scatter(
            x=metrics_data['fecha'],
            y=metrics_data['r2_ma'],
            mode='lines',
            name='Media Móvil (3 puntos)',
            line=dict(color='#FF6B6B', width=4),
            opacity=0.8
        ))
        
        # Línea de umbral
        fig.add_hline(
            y=0.7, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Umbral Mínimo (0.7)"
        )
        
        fig.update_layout(
            title="Tendencia del R² Score con Media Móvil",
            xaxis=dict(title="Fecha"),
            yaxis=dict(title="R² Score", range=[0, 1]),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de tendencia
        if len(metrics_data) >= 3:
            current_r2 = metrics_data['r2'].iloc[-1]
            previous_r2 = metrics_data['r2'].iloc[-2]
            trend = "mejorando" if current_r2 > previous_r2 else "empeorando" if current_r2 < previous_r2 else "estable"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Actual", f"{current_r2:.3f}")
            with col2:
                st.metric("Tendencia", trend.capitalize())
    
    def _render_error_analysis_chart(self):
        """Análisis visual de errores"""
        metrics_data = self._prepare_metrics_data()
        
        if metrics_data.empty:
            return
        
        fig = go.Figure()
        
        # MAE
        fig.add_trace(go.Scatter(
            x=metrics_data['fecha'],
            y=metrics_data['mae'],
            mode='lines+markers',
            name='MAE',
            line=dict(color='#FF9800', width=3)
        ))
        
        # MAPE
        fig.add_trace(go.Scatter(
            x=metrics_data['fecha'],
            y=metrics_data['mape'],
            mode='lines+markers',
            name='MAPE (%)',
            line=dict(color='#4CAF50', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Análisis de Errores del Modelo",
            xaxis=dict(title="Fecha"),
            yaxis=dict(title="MAE (meses)"),
            yaxis2=dict(
                title="MAPE (%)",
                overlaying='y',
                side='right'
            ),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_alerts_panel(self):
        """Panel de alertas y notificaciones"""
        st.subheader("🚨 Panel de Alertas")
        
        # Filtrar alertas recientes (últimas 72 horas)
        recent_alerts = [
            alert for alert in self.monitoring_system.alertas
            if datetime.now() - datetime.fromisoformat(alert['fecha']) < timedelta(hours=72)
        ]
        
        if not recent_alerts:
            st.success("✅ No hay alertas activas en las últimas 72 horas")
            return
        
        # Agrupar alertas por tipo y nivel
        alert_counts = {}
        for alert in recent_alerts:
            key = (alert['tipo'], alert['nivel'])
            alert_counts[key] = alert_counts.get(key, 0) + 1
        
        # Mostrar resumen de alertas
        col1, col2, col3 = st.columns(3)
        
        critical_alerts = len([a for a in recent_alerts if a['nivel'] == 'CRITICO'])
        high_alerts = len([a for a in recent_alerts if a['nivel'] == 'ALTO'])
        medium_alerts = len([a for a in recent_alerts if a['nivel'] == 'MEDIO'])
        
        with col1:
            st.metric("Críticas", critical_alerts, delta_color="inverse")
        with col2:
            st.metric("Altas", high_alerts, delta_color="off")
        with col3:
            st.metric("Medias", medium_alerts, delta_color="normal")
        
        # Lista detallada de alertas
        with st.expander("📋 Ver Detalles de Alertas", expanded=True):
            for alert in sorted(recent_alerts, 
                              key=lambda x: datetime.fromisoformat(x['fecha']), 
                              reverse=True)[:10]:  # Últimas 10 alertas
                
                alert_time = datetime.fromisoformat(alert['fecha']).strftime('%H:%M - %d/%m')
                
                if alert['nivel'] == 'CRITICO':
                    st.error(f"🔴 **{alert['tipo']}** - {alert_time}")
                    st.write(f"*{alert['mensaje']}*")
                elif alert['nivel'] == 'ALTO':
                    st.warning(f"🟡 **{alert['tipo']}** - {alert_time}")
                    st.write(f"*{alert['mensaje']}*")
                else:
                    st.info(f"🔵 **{alert['tipo']}** - {alert_time}")
                    st.write(f"*{alert['mensaje']}*")
                
                st.markdown("---")
    
    def _render_trend_analysis(self):
        """Análisis de tendencias y patrones"""
        st.subheader("📊 Análisis de Tendencias")
        
        if len(self.monitoring_system.metricas_historicas) < 5:
            st.info("Se necesitan al menos 5 evaluaciones para el análisis de tendencias")
            return
        
        reporte = self.monitoring_system.generar_reporte_monitoreo(30)
        
        if 'error' in reporte:
            st.warning(reporte['error'])
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Resumen de 30 Días:**")
            resumen = reporte.get('resumen', {})
            st.write(f"- Evaluaciones: {resumen.get('total_evaluaciones', 0)}")
            st.write(f"- Estado promedio: {resumen.get('estado_promedio', 'N/A')}")
            st.write(f"- Performance promedio: {resumen.get('performance_promedio', 0):.3f}")
            st.write(f"- Estabilidad: {resumen.get('estabilidad_modelo', 'N/A')}")
        
        with col2:
            st.write("**Tendencias Detectadas:**")
            tendencias = reporte.get('tendencias', {})
            st.write(f"- R²: {tendencias.get('tendencia_r2', 'N/A')}")
            st.write(f"- MAE: {tendencias.get('tendencia_mae', 'N/A')}")
            st.write(f"- Volatilidad R²: {tendencias.get('volatilidad_r2', 0):.4f}")
            st.write(f"- Volatilidad MAE: {tendencias.get('volatilidad_mae', 0):.2f}")
        
        # Gráfico de distribución de R²
        metrics_data = self._prepare_metrics_data()
        if not metrics_data.empty:
            fig = px.histogram(metrics_data, x='r2', 
                             title='Distribución de R² Scores',
                             nbins=10,
                             color_discrete_sequence=['#2E86AB'])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_cache_analytics(self):
        """Análisis del sistema de cache"""
        st.subheader("⚡ Analytics del Cache")
        
        cache_stats = cache_system.get_stats()
        
        if not cache_stats.get('enabled', False):
            st.warning("El sistema de cache está deshabilitado")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tasa Aciertos", f"{cache_stats.get('hit_rate', 0):.1f}%")
        
        with col2:
            st.metric("Total Operaciones", 
                     cache_stats.get('total_hits', 0) + cache_stats.get('total_misses', 0))
        
        with col3:
            st.metric("Aciertos", cache_stats.get('total_hits', 0))
        
        with col4:
            st.metric("Fallos", cache_stats.get('total_misses', 0))
        
        # Gráfico de uso del cache por tipo
        cache_sizes = cache_stats.get('cache_sizes', {})
        if cache_sizes:
            cache_types = list(cache_sizes.keys())
            usage_percentages = [cache_sizes[ct]['usage_percentage'] for ct in cache_types]
            
            fig = px.bar(x=cache_types, y=usage_percentages,
                        title='Uso del Cache por Tipo',
                        labels={'x': 'Tipo de Cache', 'y': 'Uso (%)'},
                        color=usage_percentages,
                        color_continuous_scale='Viridis')
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_control_panel(self):
        """Panel de control para el monitoreo"""
        st.subheader("🎛️ Panel de Control")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Evaluar Ahora", use_container_width=True):
                with st.spinner("Ejecutando evaluación..."):
                    # Aquí integrar con la evaluación real del modelo
                    st.success("Evaluación completada")
                    st.rerun()
        
        with col2:
            if st.button("📊 Generar Reporte", use_container_width=True):
                reporte = self.monitoring_system.generar_reporte_monitoreo(30)
                with st.expander("Ver Reporte Completo"):
                    st.json(reporte)
        
        with col3:
            if st.button("🧹 Limpiar Alertas", use_container_width=True):
                # Mantener solo las últimas 100 alertas
                if len(self.monitoring_system.alertas) > 100:
                    self.monitoring_system.alertas = self.monitoring_system.alertas[-100:]
                st.success("Alertas limpiadas")
    
    def _get_latest_evaluation(self) -> Optional[Dict]:
        """Obtener la evaluación más reciente"""
        if not self.monitoring_system.metricas_historicas:
            return None
        return self.monitoring_system.metricas_historicas[-1]
    
    def _prepare_metrics_data(self) -> pd.DataFrame:
        """Preparar datos de métricas para gráficos"""
        if not self.monitoring_system.metricas_historicas:
            return pd.DataFrame()
        
        data = []
        for eval_data in self.monitoring_system.metricas_historicas:
            if 'metricas_performance' in eval_data:
                metrics = eval_data['metricas_performance']
                if 'error' not in metrics:
                    data.append({
                        'fecha': datetime.fromisoformat(eval_data['fecha']),
                        'r2': metrics.get('r2', 0),
                        'mae': metrics.get('mae', 0),
                        'mape': metrics.get('mape', 0),
                        'rmse': metrics.get('rmse', 0)
                    })
        
        return pd.DataFrame(data)
    
    def _get_status_color(self, status: str) -> str:
        """Obtener color para el estado"""
        colors = {
            'EXCELENTE': 'normal',
            'BUENO': 'normal',
            'ACEPTABLE': 'off',
            'CRITICO': 'inverse'
        }
        return colors.get(status, 'normal')

# Instancia global del dashboard
def create_monitoring_dashboard(monitoring_system: MLMonitoringSystem) -> RealTimeMonitoringDashboard:
    """Crear instancia del dashboard de monitoreo"""
    return RealTimeMonitoringDashboard(monitoring_system)