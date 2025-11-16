# orthopredict_app/src/monitoring_system.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ✅ NUEVO: Importar sistemas optimizados
from ml_models import OrthoMLPredictorOptimized
from config import config
from cache_system import cache_system
import logging

logger = logging.getLogger(__name__)

class MLMonitoringSystem:
    def __init__(self):
        self.metricas_historicas = []
        self.alertas = []
        self.umbrales = {
            'degradacion_mae': 0.5,
            'degradacion_r2': 0.05,
            'drift_datos': 0.15,
            'performance_minima': 0.65,
            'cache_hit_rate_minima': 60.0  # ✅ NUEVO: Umbral para cache
        }
        self.drift_detector = DataDriftDetector()
        
    def evaluar_modelo_actual(self, modelo_actual, datos_recientes):
        """Evaluación completa con métricas de cache"""
        evaluacion = {
            'fecha': datetime.now().isoformat(),
            'metricas_calidad': self._evaluar_calidad_modelo(modelo_actual),
            'drift_datos': self.drift_detector.detectar_drift(datos_recientes),
            'metricas_performance': self._calcular_metricas_performance(modelo_actual, datos_recientes),
            'metricas_cache': cache_system.get_stats(),  # ✅ NUEVO: Métricas de cache
            'alertas_activadas': []
        }
        
        # Verificar umbrales y generar alertas
        alertas = self._verificar_umbrales(evaluacion)
        evaluacion['alertas_activadas'] = alertas
        self.alertas.extend(alertas)
        
        # Guardar histórico
        self.metricas_historicas.append(evaluacion)
        
        logger.info(f"✅ Evaluación de modelo completada - Alertas: {len(alertas)}")
        return evaluacion
    
    def _evaluar_calidad_modelo(self, modelo):
        """Evaluar calidad técnica del modelo incluyendo cache"""
        if 'principal' not in modelo.modelos:
            return {'estado': 'CRITICO', 'score': 0, 'detalles': 'Modelo no entrenado'}
        
        modelo_data = modelo.modelos['principal']
        metricas = modelo_data.get('metricas', {})
        
        score = 0
        detalles = []
        
        # Evaluar R²
        r2 = metricas.get('r2', 0)
        if r2 > 0.8:
            score += 30
            detalles.append(f"R² excelente: {r2}")
        elif r2 > 0.7:
            score += 20
            detalles.append(f"R² bueno: {r2}")
        elif r2 > 0.6:
            score += 10
            detalles.append(f"R² aceptable: {r2}")
        else:
            detalles.append(f"R² bajo: {r2}")
        
        # Evaluar MAE
        mae = metricas.get('mae', 10)
        if mae < 1.5:
            score += 25
            detalles.append(f"MAE excelente: {mae} meses")
        elif mae < 2.0:
            score += 15
            detalles.append(f"MAE bueno: {mae} meses")
        elif mae < 2.5:
            score += 10
            detalles.append(f"MAE aceptable: {mae} meses")
        else:
            detalles.append(f"MAE alto: {mae} meses")
        
        # ✅ NUEVO: Evaluar cache performance
        cache_stats = cache_system.get_stats()
        if cache_stats['enabled']:
            hit_rate = cache_stats['hit_rate']
            if hit_rate > 70:
                score += 20
                detalles.append(f"Cache excelente: {hit_rate}% hit rate")
            elif hit_rate > 50:
                score += 15
                detalles.append(f"Cache bueno: {hit_rate}% hit rate")
            elif hit_rate > 30:
                score += 10
                detalles.append(f"Cache aceptable: {hit_rate}% hit rate")
            else:
                detalles.append(f"Cache bajo: {hit_rate}% hit rate")
        
        # Evaluar antigüedad del modelo
        if modelo.ultimo_entrenamiento:
            dias_desde_entrenamiento = (datetime.now() - modelo.ultimo_entrenamiento).days
            if dias_desde_entrenamiento < 30:
                score += 15
                detalles.append(f"Modelo reciente: {dias_desde_entrenamiento} días")
            elif dias_desde_entrenamiento < 90:
                score += 10
                detalles.append(f"Modelo moderadamente antiguo: {dias_desde_entrenamiento} días")
            else:
                detalles.append(f"Modelo antiguo: {dias_desde_entrenamiento} días")
        
        # Evaluar feature importance
        if modelo.feature_importance:
            top_features = list(modelo.feature_importance.keys())[:3]
            score += 10
            detalles.append(f"Features principales: {', '.join(top_features)}")
        
        # Determinar estado
        if score >= 80:
            estado = "EXCELENTE"
        elif score >= 60:
            estado = "BUENO"
        elif score >= 40:
            estado = "ACEPTABLE"
        else:
            estado = "CRITICO"
        
        return {
            'estado': estado,
            'score': score,
            'detalles': detalles,
            'metricas_detalladas': metricas,
            'cache_stats': cache_stats  # ✅ NUEVO: Incluir stats de cache
        }
    
    def _calcular_metricas_performance(self, modelo, datos_recientes):
        """Calcular métricas de performance en datos recientes"""
        try:
            if len(datos_recientes) < 10:
                return {'error': 'Datos insuficientes para evaluación'}
            
            # Realizar predicciones en datos recientes
            predicciones = []
            valores_reales = []
            
            for _, paciente in datos_recientes.iterrows():
                datos_paciente = {
                    'edad': paciente['edad'],
                    'sexo': paciente['sexo'],
                    'apiñamiento_mm': paciente['apiñamiento_mm'],
                    'sobremordida_mm': paciente['sobremordida_mm'],
                    'sobresalte_mm': paciente['sobresalte_mm']
                }
                
                prediccion = modelo.predecir_duracion(datos_paciente)
                predicciones.append(prediccion['prediccion'])
                valores_reales.append(paciente['duracion_real_meses'])
            
            # Calcular métricas
            residuals = np.array(valores_reales) - np.array(predicciones)
            mae = np.mean(np.abs(residuals))
            mse = np.mean(residuals ** 2)
            rmse = np.sqrt(mse)
            r2 = 1 - (np.var(residuals) / np.var(valores_reales))
            mape = np.mean(np.abs(residuals / np.array(valores_reales))) * 100
            
            return {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'r2': round(r2, 3),
                'mape': round(mape, 1),
                'n_muestras': len(datos_recientes),
                'bias': round(np.mean(residuals), 2)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _verificar_umbrales(self, evaluacion):
        """Verificar si se activan alertas incluyendo cache"""
        alertas = []
        
        # Verificar degradación de performance
        metricas_performance = evaluacion['metricas_performance']
        if 'r2' in metricas_performance and metricas_performance['r2'] < self.umbrales['performance_minima']:
            alertas.append({
                'tipo': 'PERFORMANCE_BAJA',
                'nivel': 'ALTO',
                'mensaje': f'R² por debajo del umbral mínimo: {metricas_performance["r2"]}',
                'fecha': datetime.now().isoformat()
            })
        
        # ✅ NUEVO: Verificar performance del cache
        metricas_cache = evaluacion.get('metricas_cache', {})
        if metricas_cache.get('enabled', False) and 'hit_rate' in metricas_cache:
            hit_rate = metricas_cache.get('hit_rate', 0)
            if hit_rate < self.umbrales['cache_hit_rate_minima']:
                alertas.append({
                    'tipo': 'CACHE_BAJO',
                    'nivel': 'MEDIO',
                    'mensaje': f'Tasa de aciertos de cache baja: {hit_rate}%',
                    'fecha': datetime.now().isoformat()
                })
        
        # Verificar drift de datos
        drift_info = evaluacion['drift_datos']
        if drift_info['drift_detectado']:
            alertas.append({
                'tipo': 'DRIFT_DATOS',
                'nivel': 'MEDIO',
                'mensaje': f'Drift detectado en distribución de datos. Score: {drift_info["drift_score"]:.3f}',
                'fecha': datetime.now().isoformat()
            })
        
        # Verificar calidad del modelo
        calidad = evaluacion['metricas_calidad']
        if calidad['estado'] == 'CRITICO':
            alertas.append({
                'tipo': 'CALIDAD_MODELO',
                'nivel': 'CRITICO',
                'mensaje': 'Calidad del modelo en estado CRÍTICO',
                'fecha': datetime.now().isoformat()
            })
        
        return alertas
    
    def generar_reporte_monitoreo(self, periodo_dias=30):
        """Generar reporte de monitoreo con métricas de cache"""
        fecha_limite = datetime.now() - timedelta(days=periodo_dias)
        metricas_periodo = [m for m in self.metricas_historicas 
                           if datetime.fromisoformat(m['fecha']) >= fecha_limite]
        
        if not metricas_periodo:
            return {'error': 'No hay datos en el período especificado'}
        
        reporte = {
            'periodo': f"Últimos {periodo_dias} días",
            'resumen': self._generar_resumen_periodo(metricas_periodo),
            'tendencias': self._analizar_tendencias(metricas_periodo),
            'alertas_recientes': self._obtener_alertas_recientes(periodo_dias),
            'metricas_cache': cache_system.get_stats(),  # ✅ NUEVO: Métricas actuales de cache
            'recomendaciones': self._generar_recomendaciones(metricas_periodo)
        }
        
        return reporte
    
    def _generar_resumen_periodo(self, metricas_periodo):
        """Generar resumen del período de monitoreo"""
        estados = [m['metricas_calidad']['estado'] for m in metricas_periodo]
        performance_scores = [m['metricas_performance'].get('r2', 0) for m in metricas_periodo 
                             if 'r2' in m['metricas_performance']]
        
        return {
            'total_evaluaciones': len(metricas_periodo),
            'estado_promedio': max(set(estados), key=estados.count),
            'performance_promedio': round(np.mean(performance_scores), 3) if performance_scores else 0,
            'alertas_totales': len([a for a in self.alertas if datetime.fromisoformat(a['fecha']) >= 
                                   datetime.now() - timedelta(days=30)]),
            'estabilidad_modelo': self._calcular_estabilidad(metricas_periodo)
        }
    
    def _analizar_tendencias(self, metricas_periodo):
        """Analizar tendencias en las métricas"""
        if len(metricas_periodo) < 2:
            return {'error': 'Datos insuficientes para análisis de tendencias'}
        
        # Extraer series temporales
        fechas = [datetime.fromisoformat(m['fecha']) for m in metricas_periodo]
        r2_scores = [m['metricas_performance'].get('r2', 0) for m in metricas_periodo]
        mae_scores = [m['metricas_performance'].get('mae', 0) for m in metricas_periodo]
        
        # Calcular tendencias
        tendencia_r2 = self._calcular_tendencia(r2_scores)
        tendencia_mae = self._calcular_tendencia(mae_scores)
        
        return {
            'tendencia_r2': tendencia_r2,
            'tendencia_mae': tendencia_mae,
            'volatilidad_r2': round(np.std(r2_scores), 4),
            'volatilidad_mae': round(np.std(mae_scores), 2)
        }
    
    def _calcular_tendencia(self, serie):
        """Calcular tendencia de una serie temporal"""
        if len(serie) < 2:
            return 'ESTABLE'
        
        x = np.arange(len(serie))
        slope, _, _, _, _ = stats.linregress(x, serie)
        
        if slope > 0.01:
            return 'MEJORANDO'
        elif slope < -0.01:
            return 'EMPEORANDO'
        else:
            return 'ESTABLE'
    
    def _calcular_estabilidad(self, metricas_periodo):
        """Calcular estabilidad del modelo"""
        r2_scores = [m['metricas_performance'].get('r2', 0) for m in metricas_periodo 
                     if 'r2' in m['metricas_performance']]
        
        if not r2_scores or len(r2_scores) < 2:
            return 'DESCONOCIDA'
        
        cv = np.std(r2_scores) / np.mean(r2_scores)  # Coeficiente de variación
        
        if cv < 0.05:
            return 'ALTA'
        elif cv < 0.1:
            return 'MEDIA'
        else:
            return 'BAJA'
    
    def _obtener_alertas_recientes(self, periodo_dias):
        """Obtener alertas recientes"""
        fecha_limite = datetime.now() - timedelta(days=periodo_dias)
        alertas_recientes = [a for a in self.alertas 
                            if datetime.fromisoformat(a['fecha']) >= fecha_limite]
        
        return {
            'total': len(alertas_recientes),
            'por_nivel': self._contar_alertas_por_nivel(alertas_recientes),
            'recientes': alertas_recientes[-5:]  # Últimas 5 alertas
        }
    
    def _contar_alertas_por_nivel(self, alertas):
        """Contar alertas por nivel de severidad"""
        niveles = {}
        for alerta in alertas:
            nivel = alerta['nivel']
            niveles[nivel] = niveles.get(nivel, 0) + 1
        return niveles
    
    def _generar_recomendaciones(self, metricas_periodo):
        """Generar recomendaciones incluyendo optimizaciones de cache"""
        recomendaciones = []
        
        # Analizar tendencia de performance
        tendencias = self._analizar_tendencias(metricas_periodo)
        if tendencias.get('tendencia_r2') == 'EMPEORANDO':
            recomendaciones.append("🔻 El performance del modelo está empeorando. Considerar reentrenamiento.")
        
        # Verificar estabilidad
        resumen = self._generar_resumen_periodo(metricas_periodo)
        if resumen['estabilidad_modelo'] == 'BAJA':
            recomendaciones.append("⚡ Alta volatilidad en las predicciones. Investigar causas.")
        
        # ✅ NUEVO: Verificar performance del cache
        cache_stats = cache_system.get_stats()
        if cache_stats['enabled']:
            hit_rate = cache_stats.get('hit_rate', 0)
            if hit_rate < 40:
                recomendaciones.append("💾 Tasa de aciertos de cache baja. Considerar ajustar estrategia de cache.")
            elif hit_rate > 80:
                recomendaciones.append("⚡ Cache funcionando excelentemente. Buen trabajo!")
        
        # Verificar alertas recientes
        alertas_recientes = self._obtener_alertas_recientes(30)
        if alertas_recientes['total'] > 10:
            recomendaciones.append("🚨 Número elevado de alertas. Revisión urgente requerida.")
        
        # Verificar antigüedad del modelo
        if len(metricas_periodo) > 0:
            ultima_eval = metricas_periodo[-1]
            if 'metricas_calidad' in ultima_eval:
                detalles = ultima_eval['metricas_calidad'].get('detalles', [])
                for detalle in detalles:
                    if 'antiguo' in detalle.lower():
                        recomendaciones.append("🕐 Modelo antiguo. Programar reentrenamiento.")
                        break
        
        if not recomendaciones:
            recomendaciones.append("✅ El sistema se encuentra en estado óptimo. Continuar monitoreo.")
        
        return recomendaciones

class DataDriftDetector:
    """Detector de drift en la distribución de datos"""
    
    def __init__(self):
        self.distribucion_referencia = None
    
    def establecer_referencia(self, datos_referencia):
        """Establecer distribución de referencia"""
        self.distribucion_referencia = {
            'edad': self._calcular_distribucion(datos_referencia['edad']),
            'apiñamiento_mm': self._calcular_distribucion(datos_referencia['apiñamiento_mm']),
            'sobremordida_mm': self._calcular_distribucion(datos_referencia['sobremordida_mm']),
            'sobresalte_mm': self._calcular_distribucion(datos_referencia['sobresalte_mm'])
        }
    
    def detectar_drift(self, datos_actuales):
        """Detectar drift entre distribución actual y referencia"""
        if self.distribucion_referencia is None:
            return {'drift_detectado': False, 'error': 'No hay distribución de referencia'}
        
        drift_scores = {}
        
        for feature in ['edad', 'apiñamiento_mm', 'sobremordida_mm', 'sobresalte_mm']:
            if feature in datos_actuales.columns:
                distribucion_actual = self._calcular_distribucion(datos_actuales[feature])
                drift_score = self._calcular_distancia_distribuciones(
                    self.distribucion_referencia[feature], 
                    distribucion_actual
                )
                drift_scores[feature] = drift_score
        
        drift_promedio = np.mean(list(drift_scores.values())) if drift_scores else 0
        drift_detectado = drift_promedio > 0.1  # Umbral para drift
        
        return {
            'drift_detectado': drift_detectado,
            'drift_score': drift_promedio,
            'drift_por_feature': drift_scores
        }
    
    def _calcular_distribucion(self, serie):
        """Calcular distribución de una serie"""
        return {
            'mean': serie.mean(),
            'std': serie.std(),
            'min': serie.min(),
            'max': serie.max(),
            'hist': np.histogram(serie, bins=10, density=True)[0]
        }
    
    def _calcular_distancia_distribuciones(self, dist1, dist2):
        """Calcular distancia entre dos distribuciones"""
        # Distancia en medias (normalizada)
        diff_means = abs(dist1['mean'] - dist2['mean']) / (dist1['std'] + 1e-8)
        
        # Distancia en desviaciones estándar
        diff_std = abs(dist1['std'] - dist2['std']) / (dist1['std'] + 1e-8)
        
        # Distancia en histogramas (correlación)
        if len(dist1['hist']) == len(dist2['hist']):
            correlation = np.corrcoef(dist1['hist'], dist2['hist'])[0, 1]
            diff_hist = 1 - (correlation if not np.isnan(correlation) else 0)
        else:
            diff_hist = 1
        
        # Combinar métricas
        distancia_total = (diff_means + diff_std + diff_hist) / 3
        return distancia_total

# Instancia global para integración
sistema_monitoreo = MLMonitoringSystem()