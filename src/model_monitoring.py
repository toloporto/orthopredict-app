class ModelPerformanceMonitor:
    def detectar_data_drift(self, datos_actuales, datos_referencia):
        """Detectar cambios en distribución de datos"""
        from scipy import stats
        drift_metrics = {}
        for col in ['apiñamiento_mm', 'edad']:
            stat, p_value = stats.ks_2samp(
                datos_referencia[col], datos_actuales[col]
            )
            drift_metrics[col] = {'statistic': stat, 'p_value': p_value}
        return drift_metrics
    
    def calcular_metricas_degradacion(self, metricas_originales, metricas_actuales):
        """Calcular degradación del modelo"""
        mae_degradation = (
            (metricas_actuales['mae'] - metricas_originales['mae']) / 
            metricas_originales['mae']
        )
        return mae_degradation