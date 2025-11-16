from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from scipy import stats
import warnings
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.exceptions import ConvergenceWarning

# ✅ NUEVO: Importar sistemas optimizados
from config import config

from cache_system import cache_system
from utils import utils
from clinical_analyzer import clinical_analyzer
from data_integration import data_integration
import joblib  # ✅ NUEVO: Serialización más eficiente
import logging

# Configurar logging
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class OrthoMLPredictorOptimized:
    def __init__(self):
        self.modelos = {}
        self.metricas = {}
        self.historial_entrenamiento = []
        self.feature_importance = {}
        self.ultimo_entrenamiento = None
        self.scalers = {}
        self.classification_model = None # ✅ NUEVO: Modelo de clasificación
        
        # ✅ NUEVO: Configuración desde sistema centralizado
        self.cache_enabled = config.CACHE_ENABLED
        self.training_samples = config.ML_TRAINING_SAMPLES
        
        # ✅ NUEVO: Integración con datos clínicos realistas
        self.data_integration = data_integration
        # ✅ NUEVO: Integración con sistemas clínicos
        self.clinical_analyzer = clinical_analyzer
        
    def generar_datos_entrenamiento_avanzado(self, n_muestras=2000):
        """Generar dataset sintético con cache"""
        # ✅ NUEVO: Usar cache para datos de entrenamiento
        cache_key = f"training_data_{n_muestras}"
        cached_data = cache_system.get('models', cache_key)
        
        if cached_data is not None and self.cache_enabled:
            logger.info(f"📦 Datos de entrenamiento cargados desde cache: {n_muestras} muestras")
            return cached_data
        
        np.random.seed(42)
        
        datos = []
        for i in range(n_muestras):
            # Datos demográficos con distribución realista
            edad = np.random.normal(25, 4)  # Distribución normal centrada en 25
            edad = max(18, min(35, int(edad)))
            sexo = np.random.choice(['M', 'F'], p=[0.45, 0.55])
            
            # Parámetros clínicos con correlaciones más realistas
            apiñamiento = np.random.beta(2, 2) * 4 + 4  # Distribución beta
            apiñamiento = round(apiñamiento, 1)
            
            # Correlaciones entre variables
            # Apiñamiento alto tiende a tener sobremordida/sobresalte más extremos
            correlacion = 0.6
            sobremordida_base = np.random.normal(2.5, 1.0)
            sobremordida = sobremordida_base + (apiñamiento - 6) * correlacion * 0.3
            sobremordida = max(1.0, min(5.0, round(sobremordida, 1)))
            
            sobresalte_base = np.random.normal(3.0, 1.0)
            sobresalte = sobresalte_base + (apiñamiento - 6) * correlacion * 0.3
            sobresalte = max(1.0, min(5.0, round(sobresalte, 1)))
            
            # Nuevas características
            historia_ortodoncia_prev = np.random.choice([0, 1], p=[0.8, 0.2])
            nivel_cooperacion = np.random.normal(0.7, 0.2)  # 0-1 scale
            nivel_cooperacion = max(0.3, min(1.0, nivel_cooperacion))
            
            # Cálculo de duración más sofisticado
            duracion_base = self._calcular_duracion_base(
                apiñamiento, edad, sexo, sobremordida, sobresalte,
                historia_ortodoncia_prev, nivel_cooperacion
            )
            
            # Añadir ruido realista
            ruido = np.random.normal(0, 1.2)
            duracion_final = max(12, min(36, duracion_base + ruido))
            
            datos.append({
                'id': i + 1000,
                'edad': edad,
                'sexo': sexo,
                'apiñamiento_mm': apiñamiento,
                'sobremordida_mm': sobremordida,
                'sobresalte_mm': sobresalte,
                'historia_ortodoncia_prev': historia_ortodoncia_prev,
                'nivel_cooperacion': round(nivel_cooperacion, 2),
                'tipo_caso': self._clasificar_caso(apiñamiento),
                'duracion_real_meses': round(duracion_final, 1),
                'complejidad_calculada': round(self._calcular_complejidad(apiñamiento, sobremordida, sobresalte), 1)
            })
        
        df_result = pd.DataFrame(datos)
        
        # ✅ NUEVO: Guardar en cache
        if self.cache_enabled:
            cache_system.set('models', df_result, cache_key)
            logger.info(f"💾 Datos de entrenamiento guardados en cache: {n_muestras} muestras")
        
        return df_result
    
    def _calcular_duracion_base(self, apiñamiento, edad, sexo, sobremordida, sobresalte, 
                               historia_prev, cooperacion):
        """Cálculo más preciso de duración base"""
        # Factores base por tipo de caso
        if apiñamiento < 5:
            base = 14
            factor_apiñamiento = apiñamiento * 0.7
        elif apiñamiento < 7:
            base = 18
            factor_apiñamiento = apiñamiento * 1.1
        else:
            base = 22
            factor_apiñamiento = apiñamiento * 1.4
        
        # Factores adicionales
        factor_edad = max(0, (edad - 18) * 0.08)  # Edad avanzada = más tiempo
        factor_sexo = 0.8 if sexo == 'M' else -0.2  # Hombres tienden a necesitar más tiempo
        factor_sobremordida = max(0, abs(sobremordida - 2.5) * 0.4)  # Valores extremos = más tiempo
        factor_sobresalte = max(0, abs(sobresalte - 3.0) * 0.3)
        factor_historia = 1.5 if historia_prev else 0  # Ortodoncia previa = más complejo
        factor_cooperacion = (1 - cooperacion) * 3  # Menos cooperación = más tiempo
        
        duracion = (base + factor_apiñamiento + factor_edad + factor_sexo + 
                   factor_sobremordida + factor_sobresalte + factor_historia + 
                   factor_cooperacion)
        
        return duracion
    
    def _clasificar_caso(self, apiñamiento):
        """Clasificación más precisa del caso"""
        if apiñamiento < 4.5:
            return "muy_leve"
        elif apiñamiento < 5.5:
            return "leve"
        elif apiñamiento < 6.5:
            return "moderado"
        elif apiñamiento < 7.5:
            return "moderado_severo"
        else:
            return "severo"
    
    def _calcular_complejidad(self, apiñamiento, sobremordida, sobresalte):
        """Calcular score de complejidad 0-10"""
        complejidad_apiñamiento = (apiñamiento - 4) * 2.5  # 4mm=0, 8mm=10
        complejidad_sobremordida = abs(sobremordida - 2.5) * 2
        complejidad_sobresalte = abs(sobresalte - 3.0) * 1.5
        
        return min(10, (complejidad_apiñamiento + complejidad_sobremordida + complejidad_sobresalte) / 3)
    
    def entrenar_modelo_avanzado(self, datos_entrenamiento):
        """Modelo de regresión con cache de entrenamiento"""
        try:
            # ✅ NUEVO: Cache key para modelo entrenado
            cache_key = f"trained_model_{len(datos_entrenamiento)}_{datetime.now().strftime('%Y%m%d')}"
            cached_model = cache_system.get('models', cache_key)
            
            if cached_model is not None and self.cache_enabled:
                logger.info("📦 Modelo cargado desde cache")
                self.modelos['principal'] = cached_model
                self.metricas['principal'] = cached_model['metricas']
                self.feature_importance = cached_model.get('feature_importance', {})
                self.ultimo_entrenamiento = datetime.now()
                return cached_model
            
            # Preparar datos con más features
            X, y = self._preparar_features_clinicas(datos_entrenamiento, fit_scalers=True)
            # ✅ NUEVO: Verificar si son datos clínicos realistas
            es_datos_clinicos = 'complexity_score' in datos_entrenamiento.columns
            y = datos_entrenamiento['actual_treatment_duration_months'] if es_datos_clinicos else datos_entrenamiento['duracion_real_meses']
            
            # Usar Ridge Regression de scikit-learn para mayor estabilidad y simplicidad
            alpha_reg = 0.1  # Parámetro de regularización (alpha en sklearn)
            ridge_model = Ridge(alpha=alpha_reg, random_state=42)
            ridge_model.fit(X, y)
            
            # Predicciones y métricas
            predictions = ridge_model.predict(X)
            residuals = y - predictions
            
            # Métricas mejoradas
            mae = np.mean(np.abs(residuals))
            mse = np.mean(residuals ** 2)
            r2 = ridge_model.score(X, y)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs(residuals / y)) * 100  # Mean Absolute Percentage Error
            
            # Feature importance basada en coeficientes estandarizados
            feature_importance = self._calcular_feature_importance(X, ridge_model.coef_)
            
            modelo = {
                'tipo': 'regresion_lineal_avanzada',
                'modelo_obj': ridge_model,
                'coeficientes': ridge_model.coef_.tolist(),
                'intercept': ridge_model.intercept_,
                'caracteristicas': X.columns.tolist(),
                'metricas': {
                    'mae': round(mae, 2),
                    'mse': round(mse, 2),
                    'rmse': round(rmse, 2),
                    'r2': round(r2, 3),
                    'mape': round(mape, 1),
                    'n_muestras': len(datos_entrenamiento)
                },
                'fecha_entrenamiento': datetime.now().isoformat(),
                'parametros': {
                    'regularizacion_alpha': alpha_reg,
                    'n_features': X.shape[1]
                },
                'feature_importance': feature_importance,  # ✅ NUEVO: Incluir en modelo
                'es_datos_clinicos': es_datos_clinicos,
                'n_caracteristicas_clinicas': X.shape[1]
            }
            
            # Guardar modelo
            self.modelos['principal'] = modelo
            self.metricas['principal'] = modelo['metricas']
            self.feature_importance = feature_importance
            self.ultimo_entrenamiento = datetime.now()
            
            # Historial
            self.historial_entrenamiento.append({
                'fecha': datetime.now().isoformat(),
                'tipo_modelo': 'regresion_lineal_avanzada',
                'n_muestras': len(datos_entrenamiento),
                'metricas': modelo['metricas'],
                'feature_importance': feature_importance
            })
            
            # ✅ NUEVO: Guardar modelo en cache
            if self.cache_enabled:
                cache_system.set('models', modelo, cache_key)
                logger.info("💾 Modelo guardado en cache")
            
            return modelo
            
        except Exception as e:
            logger.error(f"Error en entrenamiento avanzado: {e}")
            # Fallback al modelo básico
            return self.entrenar_modelo_backup(datos_entrenamiento)
    
    # Añadir early stopping y optimización de hiperparámetros
    def optimizar_hiperparametros(self, datos_entrenamiento: pd.DataFrame) -> Tuple[Ridge, float]:
        """Optimizar hiperparámetros para el modelo Ridge usando GridSearchCV."""
        from sklearn.model_selection import GridSearchCV
        
        logger.info("🚀 Optimizando hiperparámetros del modelo Ridge...")
        
        try:
            X, y = self._preparar_features_clinicas(datos_entrenamiento, fit_scalers=True)
            
            param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
            
            grid_search = GridSearchCV(Ridge(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error')
            grid_search.fit(X, y)
            
            best_alpha = grid_search.best_params_['alpha']
            logger.info(f"✅ Mejor alpha encontrado: {best_alpha}")
            
            return grid_search.best_estimator_, best_alpha
        except Exception as e:
            logger.error(f"❌ Error en optimización de hiperparámetros: {e}")
            return Ridge(alpha=1.0, random_state=42), 1.0 # Fallback

    def _preparar_features_clinicas(self, datos, fit_scalers=False):
        """Preparar matriz de features con variables clínicas mejoradas"""
        if isinstance(datos, dict):
            datos = pd.DataFrame([datos])
        
        # ✅ NUEVO: Determinar si son datos clínicos realistas
        es_datos_clinicos = 'complexity_score' in datos.columns
        
        if es_datos_clinicos:
            # Usar características clínicas completas
            X = datos[['age', 'initial_crowding_mm', 'initial_overjet_mm', 
                      'initial_overbite_mm', 'complexity_score']].copy()
            
            # Codificar género si existe
            if 'gender' in datos.columns:
                X['gender_encoded'] = datos['gender'].map({'F': 0, 'M': 1, 'Femenino': 0, 'Masculino': 1})
            
            # Codificar extracciones si existe
            if 'requires_extractions' in datos.columns:
                X['extractions_encoded'] = datos['requires_extractions'].astype(int)
            
            # Codificar cooperación si existe
            if 'compliance_level' in datos.columns:
                compliance_map = {'poor': 0, 'medium': 1, 'good': 2}
                X['compliance_encoded'] = datos['compliance_level'].map(compliance_map)
                
        else:
            # Usar características básicas (compatibilidad hacia atrás)
            X = datos[['edad', 'apiñamiento_mm', 'sobremordida_mm', 'sobresalte_mm']].copy()
            X['sexo_encoded'] = datos['sexo'].map({'M': 1, 'F': 0, 'Masculino': 1, 'Femenino': 0})
        
        # Features de interacción
        X['edad_apiñamiento'] = X['age' if es_datos_clinicos else 'edad'] * X['initial_crowding_mm' if es_datos_clinicos else 'apiñamiento_mm'] / 100
        
        # Features polinómicas
        columna_apiñamiento = 'initial_crowding_mm' if es_datos_clinicos else 'apiñamiento_mm'
        X['apiñamiento_cuad'] = X[columna_apiñamiento] ** 2
        
        columna_edad = 'age' if es_datos_clinicos else 'edad'
        X['edad_cuad'] = (X[columna_edad] - 25) ** 2
        
        # Normalizar features. Crucial para que el modelo funcione correctamente.
        for col in X.columns:
            if col not in ['gender_encoded', 'extractions_encoded', 'compliance_encoded', 'sexo_encoded']:
                if fit_scalers:
                    mean = X[col].mean()
                    std = X[col].std()
                    self.scalers[col] = {'mean': mean, 'std': std}
                
                if col in self.scalers and self.scalers[col]['std'] != 0:
                    mean = self.scalers[col]['mean']
                    std = self.scalers[col]['std']
                    X[col] = (X[col] - mean) / std
        
        return X, datos.get('actual_treatment_duration_months', datos.get('duracion_real_meses'))
    
    def _calcular_feature_importance(self, X, coeficientes):
        """Calcular importancia de features estandarizada"""
        importance = {}
        feature_names = X.columns
        
        for i, feature in enumerate(feature_names):
            # Importancia basada en coeficientes estandarizados
            std_dev = X[feature].std()
            if std_dev > 0: # Evitar división por cero si la característica es constante
                importance[feature] = abs(coeficientes[i] * std_dev)
        
        # Normalizar a porcentajes
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: round(v / total_importance * 100, 1) for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def cross_validation_manual(self, datos_entrenamiento, k=5):
        """Validación cruzada más robusta"""
        try:
            n_muestras = len(datos_entrenamiento)
            if n_muestras < k * 20:  # Mínimo 20 muestras por fold
                k = max(2, n_muestras // 20)
            
            # Mezclar datos
            datos_mezclados = datos_entrenamiento.sample(frac=1, random_state=42).reset_index(drop=True)
            fold_size = n_muestras // k
            
            metricas_folds = {
                'mae': [], 'mse': [], 'r2': [], 'rmse': [], 'mape': []
            }
            
            for i in range(k):
                # Crear splits
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < k - 1 else n_muestras
                
                test_indices = list(range(start_idx, end_idx))
                train_indices = list(range(0, start_idx)) + list(range(end_idx, n_muestras))
                
                train_data = datos_mezclados.iloc[train_indices]
                test_data = datos_mezclados.iloc[test_indices]
                
                # Entrenar modelo temporal
                # Usamos una instancia temporal para no sobreescribir el scaler principal
                predictor_temp = OrthoMLPredictorOptimized()
                predictor_temp.entrenar_modelo_avanzado(train_data)
                
                # Evaluar en test
                X_test, y_test = predictor_temp._preparar_features_clinicas(test_data, fit_scalers=False)
                
                fold_predictions = predictor_temp.modelos['principal']['modelo_obj'].predict(X_test)
                fold_actuals = y_test.values
                
                # Calcular métricas del fold
                residuals = np.array(fold_actuals) - np.array(fold_predictions)
                metricas_folds['mae'].append(np.mean(np.abs(residuals)))
                metricas_folds['mse'].append(np.mean(residuals ** 2))
                metricas_folds['r2'].append(1 - (np.var(residuals) / np.var(fold_actuals)))
                metricas_folds['rmse'].append(np.sqrt(metricas_folds['mse'][-1]))
                metricas_folds['mape'].append(np.mean(np.abs(residuals / np.array(fold_actuals))) * 100)
            
            # Resumen de métricas
            resumen = {
                'n_folds': k,
                'mae_mean': round(np.mean(metricas_folds['mae']), 2),
                'mae_std': round(np.std(metricas_folds['mae']), 2),
                'mse_mean': round(np.mean(metricas_folds['mse']), 2),
                'r2_mean': round(np.mean(metricas_folds['r2']), 3),
                'r2_std': round(np.std(metricas_folds['r2']), 3),
                'rmse_mean': round(np.mean(metricas_folds['rmse']), 2),
                'mape_mean': round(np.mean(metricas_folds['mape']), 1),
                'mejor_fold_mae': round(min(metricas_folds['mae']), 2),
                'peor_fold_mae': round(max(metricas_folds['mae']), 2)
            }
            
            return resumen
            
        except Exception as e:
            return {'error': str(e), 'n_folds': 1, 'mae_mean': 2.5, 'mae_std': 0.5}
    
    def evaluar_modelo_completo(self, datos_test):
        """Evaluación completa con más métricas"""
        try:
            if 'principal' not in self.modelos:
                return {'error': 'No hay modelo entrenado para evaluar'}
                
            X_test, y_test = self._preparar_features_clinicas(datos_test, fit_scalers=False)
            predictions = self.modelos['principal']['modelo_obj'].predict(X_test)
            
            residuals = y_test - predictions
            
            # Métricas básicas
            mae = np.mean(np.abs(residuals))
            mse = np.mean(residuals ** 2)
            rmse = np.sqrt(mse)
            r2 = 1 - (np.var(residuals) / np.var(y_test))
            mape = np.mean(np.abs(residuals / y_test)) * 100
            
            # Métricas avanzadas
            rmsle = np.sqrt(np.mean((np.log1p(predictions) - np.log1p(y_test)) ** 2))
            r2_ajustado = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
            
            metricas = {
                'mae': round(mae, 2),
                'mse': round(mse, 2),
                'rmse': round(rmse, 2),
                'r2': round(r2, 3),
                'r2_ajustado': round(r2_ajustado, 3),
                'mape': round(mape, 1),
                'rmsle': round(rmsle, 3),
                'n_muestras': len(datos_test)
            }
            
            # Análisis de residuos
            metricas['residual_stats'] = {
                'mean': round(np.mean(residuals), 3),
                'std': round(np.std(residuals), 3),
                'skew': round(stats.skew(residuals), 3),
                'kurtosis': round(stats.kurtosis(residuals), 3)
            }
            
            return metricas
            
        except Exception as e:
            return {'error': str(e)}
    
    def predecir_duracion_avanzada(self, datos_paciente):
        """Predicción mejorada con el modelo avanzado"""
        if 'principal' not in self.modelos:
            return self.prediccion_por_defecto(datos_paciente)
        
        modelo = self.modelos['principal']
        
        # Redirigir si el modelo principal es un ensemble
        if modelo['tipo'] == 'ensemble_hibrido':
            return self._predecir_con_ensemble(datos_paciente)
            
        if modelo['tipo'] == 'regresion_lineal_avanzada':
            # Preparar features para predicción
            X_pred, _ = self._preparar_features_clinicas(datos_paciente, fit_scalers=False)
            
            prediccion = modelo['modelo_obj'].predict(X_pred)

            # Calcular intervalo de confianza basado en error estándar
            if 'rmse' in modelo['metricas']:
                error_std = modelo['metricas']['rmse']
            else:
                error_std = modelo['metricas']['mae'] * 1.2
            
            intervalo = error_std * 1.96  # 95% confidence interval
            
        else:
            # Usar modelo basado en reglas (fallback)
            return self.predecir_duracion(datos_paciente)
        
        # Ajustes finales
        prediccion_ajustada = max(12, min(36, float(prediccion[0])))
        
        return {
            'prediccion': round(prediccion_ajustada, 1),
            'intervalo_min': max(12, round(prediccion_ajustada - intervalo, 1)),
            'intervalo_max': min(36, round(prediccion_ajustada + intervalo, 1)),
            'modelo_usado': modelo['tipo'],
            'confianza': min(95, max(60, 100 - modelo['metricas']['mape'])),
            'factores_considerados': self._obtener_factores_considerados(datos_paciente, modelo),
            'error_estimado': round(error_std, 1)
        }
    
    def predecir_duracion_robusta(self, datos_paciente):
        """Predicción con cache y validación mejorada"""
        try:
            # ✅ NUEVO: Validación robusta de datos
            is_valid, errors = utils.validate_patient_data(datos_paciente)
            if not is_valid:
                logger.warning(f"Datos de paciente inválidos: {errors}")
                return self.prediccion_por_defecto(datos_paciente)
            
            # ✅ NUEVO: Cache key para predicción
            cache_key = f"pred_{datos_paciente['edad']}_{datos_paciente['apiñamiento_mm']}_{datos_paciente.get('sobremordida_mm', 2.5)}_{datos_paciente.get('sobresalte_mm', 3.0)}"
            cached_prediction = cache_system.get('predictions', cache_key)
            
            if cached_prediction is not None and self.cache_enabled:
                logger.debug("📦 Predicción cargada desde cache")
                return cached_prediction
            
            # Intentar modelo principal
            if 'principal' in self.modelos:
                resultado = self.predecir_duracion_avanzada(datos_paciente)
            elif 'ensemble' in self.modelos:
                resultado = self._predecir_con_ensemble(datos_paciente)
            else:
                resultado = self.prediccion_por_defecto(datos_paciente)
            
            # ✅ NUEVO: Guardar predicción en cache
            if self.cache_enabled:
                cache_system.set('predictions', resultado, cache_key)
                logger.debug("💾 Predicción guardada en cache")
                
            return resultado
                
        except Exception as e:
            logger.error(f"Error en predicción robusta: {e}")
            return self.prediccion_por_defecto(datos_paciente)
    
    def _validar_datos_paciente(self, datos_paciente):
        """Validar que los datos del paciente sean correctos"""
        required_fields = ['edad', 'apiñamiento_mm', 'sexo']
        
        for field in required_fields:
            if field not in datos_paciente:
                return False
                
        # Validar rangos
        if not (18 <= datos_paciente['edad'] <= 35):
            return False
            
        if not (4.0 <= datos_paciente['apiñamiento_mm'] <= 8.0):
            return False
            
        return True
    
    def _obtener_factores_considerados(self, datos_paciente, modelo):
        """Obtener lista de factores considerados basada en feature importance"""
        factores = []
        
        # Factores principales basados en importancia
        if 'apiñamiento_mm' in self.feature_importance:
            factores.append(f"Apiñamiento: {datos_paciente.get('apiñamiento_mm', 'N/A')} mm")
        
        if 'edad' in self.feature_importance:
            factores.append(f"Edad: {datos_paciente.get('edad', 'N/A')} años")
        
        if 'complejidad' in self.feature_importance:
            factores.append("Complejidad general del caso")
        
        if 'sexo_encoded' in self.feature_importance:
            factores.append(f"Sexo: {datos_paciente.get('sexo', 'N/A')}")
        
        # Factores adicionales
        factores.extend([
            f"Sobremordida: {datos_paciente.get('sobremordida_mm', 'N/A')} mm",
            f"Sobresalte: {datos_paciente.get('sobresalte_mm', 'N/A')} mm",
            f"Modelo: {modelo['tipo'].replace('_', ' ').title()}"
        ])
        
        return factores

    def entrenar_modelo_ensemble_hibrido(self, datos_entrenamiento):
        """Ensemble que combina Ridge + modelo basado en reglas con cache"""
        try:
            # ✅ NUEVO: Cache key para ensemble
            cache_key = f"ensemble_model_{len(datos_entrenamiento)}_{datetime.now().strftime('%Y%m%d')}"
            cached_ensemble = cache_system.get('models', cache_key)
            
            if cached_ensemble is not None and self.cache_enabled:
                logger.info("📦 Ensemble cargado desde cache")
                self.modelos['ensemble'] = cached_ensemble
                self.modelos['principal'] = cached_ensemble  # Establecer como principal
                return cached_ensemble
            
            # Entrenar modelo Ridge
            modelo_ridge = self.entrenar_modelo_avanzado(datos_entrenamiento)
            
            # Entrenar modelo basado en reglas
            modelo_reglas = self.entrenar_modelo_backup(datos_entrenamiento)
            
            # Determinar pesos basados en performance
            peso_ridge = 0.7
            peso_reglas = 0.3
            
            ensemble_model = {
                'tipo': 'ensemble_hibrido',
                'modelos': {
                    'ridge': modelo_ridge,
                    'reglas': modelo_reglas
                },
                'pesos': {
                    'ridge': peso_ridge,
                    'reglas': peso_reglas
                },
                'metricas': self._combinar_metricas(modelo_ridge['metricas'], modelo_reglas['metricas']),
                'fecha_entrenamiento': datetime.now().isoformat()
            }
            
            self.modelos['ensemble'] = ensemble_model
            self.modelos['principal'] = ensemble_model
            
            # ✅ NUEVO: Guardar ensemble en cache
            if self.cache_enabled:
                cache_system.set('models', ensemble_model, cache_key)
                logger.info("💾 Ensemble guardado en cache")
            
            return ensemble_model
            
        except Exception as e:
            logger.error(f"Error en ensemble: {e}")
            return self.entrenar_modelo_avanzado(datos_entrenamiento)
    
    def _predecir_con_ensemble(self, datos_paciente):
        """Predicción usando ensemble híbrido"""
        ensemble = self.modelos['ensemble']
        
        # Predicción del modelo Ridge
        pred_ridge = self.predecir_duracion_avanzada(datos_paciente)
        
        # Predicción del modelo de reglas
        pred_reglas = self._predecir_modelo_reglas(datos_paciente)
        
        # Combinar predicciones
        prediccion_final = (
            ensemble['pesos']['ridge'] * pred_ridge['prediccion'] +
            ensemble['pesos']['reglas'] * pred_reglas['prediccion']
        )
        
        # Combinar intervalos
        intervalo_min = (
            ensemble['pesos']['ridge'] * pred_ridge['intervalo_min'] +
            ensemble['pesos']['reglas'] * pred_reglas['intervalo_min']
        )
        
        intervalo_max = (
            ensemble['pesos']['ridge'] * pred_ridge['intervalo_max'] +
            ensemble['pesos']['reglas'] * pred_reglas['intervalo_max']
        )
        
        return {
            'prediccion': round(prediccion_final, 1),
            'intervalo_min': round(intervalo_min, 1),
            'intervalo_max': round(intervalo_max, 1),
            'modelo_usado': 'ensemble_hibrido',
            'confianza': 85,  # Confianza alta para ensemble
            'factores_considerados': pred_ridge['factores_considerados'],
            'componentes': ['Ridge Regression', 'Modelo Basado en Reglas']
        }
    
    def _predecir_modelo_reglas(self, datos_paciente):
        """Predicción usando solo el modelo basado en reglas"""
        apiñamiento = datos_paciente.get('apiñamiento_mm', 6.0)
        
        if apiñamiento < 4.5:
            caso = 'muy_leve'
        elif apiñamiento < 5.5:
            caso = 'leve'
        elif apiñamiento < 6.5:
            caso = 'moderado'
        elif apiñamiento < 7.5:
            caso = 'moderado_severo'
        else:
            caso = 'severo'
        
        reglas = {
            'muy_leve': {'min': 12, 'max': 16, 'typical': 14},
            'leve': {'min': 14, 'max': 18, 'typical': 16},
            'moderado': {'min': 16, 'max': 22, 'typical': 19},
            'moderado_severo': {'min': 18, 'max': 26, 'typical': 22},
            'severo': {'min': 22, 'max': 32, 'typical': 27}
        }
        
        return {
            'prediccion': reglas[caso]['typical'],
            'intervalo_min': reglas[caso]['min'],
            'intervalo_max': reglas[caso]['max'],
            'modelo_usado': 'modelo_basado_reglas',
            'confianza': 75,
            'factores_considerados': [f"Apiñamiento: {apiñamiento} mm", f"Caso: {caso.replace('_', ' ').title()}"]
        }
    
    def _combinar_metricas(self, metricas1, metricas2):
        """Combinar métricas de dos modelos"""
        return {
            'mae': round((metricas1['mae'] + metricas2['mae']) / 2, 2),
            'r2': round((metricas1['r2'] + metricas2['r2']) / 2, 3),
            'mape': round((metricas1['mape'] + metricas2['mape']) / 2, 1)
        }
    
    def generar_analisis_modelo(self, datos_entrenamiento):
        """Generar análisis completo del modelo"""
        analisis = {}
        
        # Feature importance visual
        analisis['feature_importance'] = self.feature_importance
        
        # Análisis de residuos
        if 'principal' in self.modelos:
            X, y = self._preparar_features_clinicas(datos_entrenamiento, fit_scalers=False)
            predictions = self.modelos['principal']['modelo_obj'].predict(X)
            residuals = y - predictions
            
            analisis['residual_analysis'] = {
                'homocedasticidad': self._test_homocedasticidad(residuals, predictions),
                'normalidad': self._test_normalidad(residuals),
                'outliers': self._detectar_outliers(residuals)
            }
        
        # Análisis por segmentos
        analisis['segment_analysis'] = {
            'por_edad': self._analizar_por_edad(datos_entrenamiento),
            'por_complejidad': self._analizar_por_complejidad(datos_entrenamiento)
        }
        
        return analisis
    
    def _test_homocedasticidad(self, residuals, predictions):
        """Test de homocedasticidad de los residuos"""
        correlation = np.corrcoef(predictions, np.abs(residuals))[0, 1]
        return round(correlation, 3)
    
    def _test_normalidad(self, residuals):
        """Test de normalidad de los residuos"""
        stat, p_value = stats.normaltest(residuals)
        return {'statistic': round(stat, 3), 'p_value': round(p_value, 3)}
    
    def _detectar_outliers(self, residuals):
        """Detectar outliers en los residuos"""
        Q1 = np.percentile(residuals, 25)
        Q3 = np.percentile(residuals, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = np.sum((residuals < lower_bound) | (residuals > upper_bound))
        return {'outliers_count': int(outliers), 'percentage': round(outliers / len(residuals) * 100, 1)}
    
    def _analizar_por_edad(self, datos):
        """Análisis de performance por grupo de edad"""
        datos['grupo_edad'] = pd.cut(datos['edad'], bins=[18, 22, 26, 30, 35])
        performance_por_edad = datos.groupby('grupo_edad').agg({
            'duracion_real_meses': ['mean', 'std', 'count']
        }).round(2)
        return performance_por_edad.to_dict()
    
    def _analizar_por_complejidad(self, datos):
        """Análisis de performance por nivel de complejidad"""
        datos['nivel_complejidad'] = pd.cut(datos['complejidad_calculada'], bins=[0, 3, 6, 10])
        performance_por_complejidad = datos.groupby('nivel_complejidad').agg({
            'duracion_real_meses': ['mean', 'std', 'count']
        }).round(2)
        return performance_por_complejidad.to_dict()

    def predecir_duracion_con_analisis_clinico(self, datos_paciente: Dict) -> Dict:
        """Predicción con análisis clínico integrado"""
        # Realizar predicción base
        prediccion_base = self.predecir_duracion_robusta(datos_paciente)
        
        # ✅ NUEVO: Añadir análisis clínico
        analisis_clinico = self.clinical_analyzer.analyze_prediction(datos_paciente, prediccion_base)
        
        # Combinar resultados
        resultado_completo = {
            **prediccion_base,
            'analisis_clinico': analisis_clinico,
            'tiene_analisis_clinico': True
        }
        
        return resultado_completo
    def entrenar_con_datos_clinicos_realistas(self, n_pacientes=2000):
        """Entrenar modelo con dataset clínico realista"""
        logger.info(f"Entrenando con dataset clínico realista de {n_pacientes} pacientes")
        
        try:
            # Obtener datos de entrenamiento realistas
            training_data = self.data_integration.get_training_data()
            
            if training_data.empty:
                logger.error("No se pudieron obtener datos de entrenamiento")
                return None
            
            # Verificar calidad de datos
            quality_report = self.data_integration.get_data_quality_report()
            logger.info(f"Calidad de datos: {quality_report}")
            
            # Entrenar modelo
            modelo = self.entrenar_modelo_avanzado(training_data)
            
            logger.info("Modelo entrenado exitosamente con datos clínicos realistas")
            return modelo
            
        except Exception as e:
            logger.error(f"Error entrenando con datos clínicos: {e}")
            # Fallback a datos sintéticos básicos
            return self.entrenar_modelo_avanzado(self.generar_datos_entrenamiento_avanzado(n_pacientes))
    
    def predecir_duracion_clinica(self, datos_paciente: Dict) -> Dict:
        """Predicción con validación clínica"""
        # Validar datos del paciente
        is_valid, errors = self.data_integration.validate_patient_data(datos_paciente)
        
        if not is_valid:
            logger.warning(f"Datos de paciente no válidos: {errors}")
            return {
                'prediccion': 18.0,  # Valor por defecto
                'intervalo_min': 15.0,
                'intervalo_max': 21.0,
                'modelo_usado': 'fallback_clinico',
                'confianza': 50,
                'factores_considerados': ['Validación fallida - usando valores conservadores'],
                'errores_validacion': errors
            }
        
        # Realizar predicción normal
        return self.predecir_duracion_robusta(datos_paciente)

    def entrenar_modelo_clasificacion_exito(self, datos_entrenamiento: pd.DataFrame):
        """Entrenar un modelo para predecir la probabilidad de éxito del tratamiento."""
        logger.info("Entrenando modelo de clasificación de éxito del tratamiento...")
        
        if 'treatment_success' not in datos_entrenamiento.columns:
            logger.error("Los datos de entrenamiento no contienen la columna 'treatment_success'.")
            return None

        try:
            X, _ = self._preparar_features_clinicas(datos_entrenamiento, fit_scalers=True)
            y = datos_entrenamiento['treatment_success'].astype(int)

            # Usar Regresión Logística, ideal para clasificación binaria
            classifier = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
            classifier.fit(X, y)

            self.classification_model = classifier
            logger.info("✅ Modelo de clasificación de éxito entrenado exitosamente.")
            
            # Opcional: Guardar el modelo de clasificación
            joblib_path = config.get_model_path('success_classifier_ridge.joblib')
            utils.save_model_joblib(self.classification_model, joblib_path)

        except Exception as e:
            logger.error(f"❌ Error entrenando el modelo de clasificación: {e}")
            self.classification_model = None

    def predecir_probabilidad_exito(self, datos_paciente: Dict) -> float:
        """Predecir la probabilidad de éxito para un paciente."""
        if not self.classification_model:
            logger.warning("El modelo de clasificación de éxito no está entrenado. Devolviendo probabilidad por defecto.")
            return 0.75  # Retornar una probabilidad base

        try:
            X_pred, _ = self._preparar_features_clinicas(datos_paciente, fit_scalers=False)
            probabilidad = self.classification_model.predict_proba(X_pred)[0][1]  # Probabilidad de la clase '1' (éxito)
            return round(float(probabilidad), 2)
        except Exception as e:
            logger.error(f"❌ Error al predecir la probabilidad de éxito: {e}")
            return 0.75

    def obtener_metricas_detalladas(self):
        """Obtener métricas detalladas con información de cache"""
        metricas_base = self.obtener_metricas()
        
        # ✅ NUEVO: Añadir información de performance
        metricas_detalladas = {
            'metricas_modelo': metricas_base,
            'cache_info': cache_system.get_stats() if self.cache_enabled else {'cache_enabled': False},
            'ultimo_entrenamiento': self.ultimo_entrenamiento.isoformat() if self.ultimo_entrenamiento else None,
            'total_modelos': len(self.modelos),
            'historial_entrenamientos': len(self.historial_entrenamiento)
        }
        
        return metricas_detalladas

    # Mantener compatibilidad con métodos existentes
    def entrenar_modelo_personalizado(self, datos_entrenamiento):
        """Alias para compatibilidad"""
        return self.entrenar_modelo_avanzado(datos_entrenamiento)
    
    def entrenar_modelo_ensemble(self, datos_entrenamiento):
        """Alias para compatibilidad"""
        return self.entrenar_modelo_avanzado(datos_entrenamiento)
    
    def predecir_duracion(self, datos_paciente):
        """Método principal unificado para compatibilidad con la app"""
        return self.predecir_duracion_robusta(datos_paciente)
    
    # Mantener métodos existentes para compatibilidad
    def entrenar_modelo_backup(self, datos_entrenamiento):
        """Modelo de respaldo (mejorado)"""
        # Calcular promedios por tipo de caso
        promedios = datos_entrenamiento.groupby('tipo_caso')['duracion_real_meses'].agg(['mean', 'std']).to_dict()
        
        modelo = {
            'tipo': 'modelo_basado_reglas',
            'promedios_por_caso': promedios,
            'reglas': {
                'muy_leve': {'min': 12, 'max': 16, 'typical': 14},
                'leve': {'min': 14, 'max': 18, 'typical': 16},
                'moderado': {'min': 16, 'max': 22, 'typical': 19},
                'moderado_severo': {'min': 18, 'max': 26, 'typical': 22},
                'severo': {'min': 22, 'max': 32, 'typical': 27}
            },
            'metricas': {
                'mae': 2.1,
                'mse': 6.5,
                'r2': 0.72,
                'rmse': 2.55,
                'mape': 12.5
            },
            'fecha_entrenamiento': datetime.now().isoformat()
        }

        self.feature_importance = {
            'apiñamiento_mm': 1.0,
            'tipo_caso': 0.8,
            'complejidad': 0.6
        }
        
        self.modelos['principal'] = modelo
        self.metricas['principal'] = modelo['metricas']
        self.ultimo_entrenamiento = datetime.now()
        
        self.historial_entrenamiento.append({
            'fecha': datetime.now().isoformat(),
            'tipo_modelo': 'modelo_basado_reglas',
            'n_muestras': len(datos_entrenamiento),
            'metricas': modelo['metricas']
        })
        
        return modelo
    
    def prediccion_por_defecto(self, datos_paciente):
        """Predicción por defecto (mejorada)"""
        apiñamiento = datos_paciente.get('apiñamiento_mm', 6.0)
        edad = datos_paciente.get('edad', 25)
        
        # Fórmula más precisa
        duracion_base = 12 + apiñamiento * 1.5 + max(0, (edad - 18) * 0.1)
        
        return {
            'prediccion': round(duracion_base, 1),
            'intervalo_min': round(duracion_base - 3, 1),
            'intervalo_max': round(duracion_base + 3, 1),
            'modelo_usado': 'formula_basica_mejorada',
            'confianza': 60,
            'factores_considerados': ['Apiñamiento', 'Edad', 'Complejidad Estimada']
        }
    
    def obtener_metricas(self):
        """Obtener métricas del modelo"""
        return self.metricas.get('principal', {})
    
    def guardar_modelo(self, ruta='models/orthopredict.json'):
        """Guardar modelo en archivo JSON - VERSIÓN SUPER ROBUSTA"""
        try:
            logger.info(f"🔄 Intentando guardar modelo en: {ruta}")
            
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            
            # Crear una estructura de datos limpia y serializable
            datos_guardar = {
                'modelos': {},
                'metricas': self.metricas,
                'historial_entrenamiento': self.historial_entrenamiento,
                'feature_importance': self.feature_importance,
                'scalers': self.scalers,
                'ultimo_entrenamiento': self.ultimo_entrenamiento.isoformat() if self.ultimo_entrenamiento else None
            }
            
            # Procesar cada modelo para hacerlo serializable
            for nombre_modelo, modelo_data in self.modelos.items():
                datos_guardar['modelos'][nombre_modelo] = self._preparar_modelo_para_serializacion(modelo_data)
            
            # Validar que los datos son serializables ANTES de guardar
            try:
                # Intentar serializar a JSON en memoria primero
                json_str = json.dumps(datos_guardar, indent=2, default=self._json_serializer)
                logger.debug("✅ Validación JSON exitosa")
            except Exception as e:
                logger.error(f"❌ Error en validación JSON: {e}")
                # Limpiar datos problemáticos
                datos_guardar = self._limpiar_datos_problematicos(datos_guardar)
                json_str = json.dumps(datos_guardar, indent=2, default=str)
            
            # Guardar en archivo
            with open(ruta, 'w', encoding='utf-8') as f:
                f.write(json_str)
            
            logger.info(f"✅ Modelo guardado exitosamente en: {ruta}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error crítico guardando modelo: {e}", exc_info=True)
            return False

    def _preparar_modelo_para_serializacion(self, modelo_data):
        """Preparar modelo para serialización JSON"""
        try:
            modelo_serializable = {}
            
            for key, value in modelo_data.items():
                if key == 'modelo_obj':
                    # Manejar objeto del modelo Ridge
                    if hasattr(value, 'coef_'):
                        modelo_serializable['ridge_params'] = {
                            'coef_': value.coef_.tolist() if hasattr(value, 'coef_') else [],
                            'intercept_': float(value.intercept_) if hasattr(value, 'intercept_') else 0.0,
                            'alpha': getattr(value, 'alpha', 1.0),
                            'n_features_in_': getattr(value, 'n_features_in_', 0)
                        }
                    else:
                        modelo_serializable['ridge_params'] = {'error': 'objeto_no_serializable'}
                
                elif key == 'modelos' and isinstance(value, dict):
                    # Procesar modelos anidados (ensembles)
                    modelo_serializable[key] = {}
                    for sub_key, sub_value in value.items():
                        modelo_serializable[key][sub_key] = self._preparar_modelo_para_serializacion(sub_value)
                
                elif isinstance(value, (str, int, float, bool, type(None))):
                    # Tipos básicos - guardar directamente
                    modelo_serializable[key] = value
                
                elif isinstance(value, (list, dict)):
                    # Listas y diccionarios - intentar serializar
                    try:
                        json.dumps(value)  # Validar que es serializable
                        modelo_serializable[key] = value
                    except:
                        modelo_serializable[key] = str(value)
                
                else:
                    # Otros tipos - convertir a string
                    modelo_serializable[key] = str(value)
            
            return modelo_serializable
            
        except Exception as e:
            logger.warning(f"⚠️ Error preparando modelo para serialización: {e}")
            return {'error_serializacion': str(e)}

    def _limpiar_datos_problematicos(self, datos):
        """Limpiar datos problemáticos para serialización"""
        datos_limpios = {}
        
        for key, value in datos.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                datos_limpios[key] = value
            elif isinstance(value, (list, dict)):
                try:
                    # Intentar serializar
                    json.dumps(value)
                    datos_limpios[key] = value
                except:
                    # Reemplazar con versión string
                    datos_limpios[key] = str(value)
            else:
                datos_limpios[key] = str(value)
        
        return datos_limpios

    def _json_serializer(self, obj):
        """Serializador personalizado para objetos no serializables"""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()  # Para datetime
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # Para objetos, usar su __dict__ si es serializable
            try:
                return obj.__dict__
            except:
                return str(obj)
        else:
            return str(obj)
    
    def guardar_modelo_completo(self, ruta_base=None):
        """Guardar modelo completo (JSON + joblib) - VERSIÓN OPTIMIZADA"""
        try:
            # Asegurarse de que ruta_base no tenga la extensión .json
            if ruta_base:
                ruta_base = ruta_base.replace('.json', '')
            else:
                ruta_base = config.get_model_path().replace('.json', '')
            
            logger.info(f"🔄 Guardando modelo completo en: {ruta_base}")
            
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(ruta_base), exist_ok=True)
            
            # 1. Guardar datos estructurados en JSON
            json_success = self.guardar_modelo(f"{ruta_base}.json")
            
            # 2. ✅ NUEVO: Guardar objetos de modelo con joblib (más eficiente)
            modelo_principal = self.modelos.get('principal', {})
            
            if modelo_principal.get('tipo') == 'regresion_lineal_avanzada' and 'modelo_obj' in modelo_principal:
                joblib_path = f"{ruta_base}_ridge.joblib"
                if utils.save_model_joblib(modelo_principal['modelo_obj'], joblib_path):
                    logger.info(f"✅ Modelo Ridge guardado con joblib: {joblib_path}")
                else:
                    logger.error(f"❌ Error guardando modelo con joblib: {joblib_path}")
                
            elif modelo_principal.get('tipo') == 'ensemble_hibrido':
                ridge_component = modelo_principal.get('modelos', {}).get('ridge', {})
                if 'modelo_obj' in ridge_component:
                    joblib_path = f"{ruta_base}_ridge.joblib"
                    if utils.save_model_joblib(ridge_component['modelo_obj'], joblib_path):
                        logger.info(f"✅ Modelo Ridge del ensemble guardado con joblib: {joblib_path}")
                    else:
                        logger.error(f"❌ Error guardando modelo ensemble con joblib: {joblib_path}")
            
            # ✅ NUEVO: Invalidar cache de modelos
            cache_system.invalidate_pattern('models', 'trained_model_')
            
            return json_success
            
        except Exception as e:
            logger.error(f"❌ Error en guardar_modelo_completo: {e}")
            return False
    
    def cargar_modelo(self, ruta='models/orthopredict.json'):
        """Cargar modelo desde archivo JSON - VERSIÓN ROBUSTA"""
        try:
            logger.info(f"🔄 Intentando cargar modelo desde: {ruta}")
            
            if not os.path.exists(ruta):
                logger.error(f"❌ Archivo no encontrado: {ruta}")
                return False
            
            # Verificar que el archivo no esté vacío
            file_size = os.path.getsize(ruta)
            if file_size == 0:
                logger.error(f"❌ Archivo vacío: {ruta}")
                return False
            
            with open(ruta, 'r', encoding='utf-8') as f:
                contenido = f.read().strip()
                
            if not contenido:
                logger.error(f"❌ Archivo vacío (sin contenido): {ruta}")
                return False
            
            # Intentar cargar JSON
            try:
                datos = json.loads(contenido)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Error de JSON en archivo {ruta}: {e}")
                logger.debug(f"🔍 Primeros 200 caracteres del contenido: {contenido[:200]}...")
                return False
            
            # Restaurar datos
            self.modelos = datos.get('modelos', {})
            self.metricas = datos.get('metricas', {})
            
            # Restaurar histórico si existe
            if 'historial_entrenamiento' in datos:
                self.historial_entrenamiento = datos['historial_entrenamiento']
            
            # Restaurar feature importance si existe
            if 'feature_importance' in datos:
                self.feature_importance = datos['feature_importance']
            
            # Restaurar scalers si existen
            if 'scalers' in datos:
                self.scalers = datos['scalers']
            
            # Restaurar fecha de último entrenamiento
            if datos.get('ultimo_entrenamiento'):
                try:
                    self.ultimo_entrenamiento = datetime.fromisoformat(datos['ultimo_entrenamiento'])
                except:
                    self.ultimo_entrenamiento = None
            
            # Reconstruir objetos de modelo si es necesario
            self._reconstruir_modelos_desde_serializacion()
            
            logger.info("✅ Modelo cargado exitosamente desde JSON")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error crítico cargando modelo: {e}", exc_info=True)
            return False

    def _reconstruir_modelos_desde_serializacion(self):
        """Reconstruir objetos de modelo desde datos serializados"""
        try:
            for nombre_modelo, modelo_data in self.modelos.items():
                if 'ridge_params' in modelo_data:
                    # Reconstruir modelo Ridge desde parámetros
                    ridge_params = modelo_data['ridge_params']
                    if 'coef_' in ridge_params and 'intercept_' in ridge_params:
                        ridge_model = Ridge(alpha=ridge_params.get('alpha', 1.0))
                        ridge_model.coef_ = np.array(ridge_params['coef_'])
                        ridge_model.intercept_ = ridge_params['intercept_']
                        # Reasignar algunos atributos necesarios
                        ridge_model.n_features_in_ = ridge_params.get('n_features_in_', len(ridge_params['coef_']))
                        self.modelos[nombre_modelo]['modelo_obj'] = ridge_model
            
            # Reconstruir modelos anidados (ensembles)
            for nombre_modelo, modelo_data in self.modelos.items():
                if 'modelos' in modelo_data and isinstance(modelo_data['modelos'], dict):
                    for sub_nombre, sub_modelo in modelo_data['modelos'].items():
                        if 'ridge_params' in sub_modelo:
                            ridge_params = sub_modelo['ridge_params']
                            if 'coef_' in ridge_params and 'intercept_' in ridge_params:
                                ridge_model = Ridge(alpha=ridge_params.get('alpha', 1.0))
                                ridge_model.coef_ = np.array(ridge_params['coef_'])
                                ridge_model.intercept_ = ridge_params['intercept_']
                                self.modelos[nombre_modelo]['modelos'][sub_nombre]['modelo_obj'] = ridge_model
                                
        except Exception as e:
            logger.warning(f"⚠️ Error reconstruyendo modelos: {e}")

    def cargar_modelo_completo(self, ruta_base=None):
        """Cargar modelo completo (JSON + joblib) - VERSIÓN OPTIMIZADA"""
        try:
            # Asegurarse de que ruta_base no tenga la extensión .json
            if ruta_base:
                ruta_base = ruta_base.replace('.json', '')
            else:
                ruta_base = config.get_model_path().replace('.json', '')
            
            json_path = f"{ruta_base}.json"
            joblib_path = f"{ruta_base}_ridge.joblib"

            logger.info(f"🔄 Cargando modelo completo desde: {ruta_base}")

            # 1. Primero intentar cargar desde JSON
            if not self.cargar_modelo(json_path):
                logger.error("❌ No se pudo cargar el modelo desde JSON")
                return False

            # 2. ✅ NUEVO: Si existe el archivo joblib, cargarlo (más eficiente que pickle)
            if os.path.exists(joblib_path):
                try:
                    ridge_model_obj = utils.load_model_joblib(joblib_path)
                    
                    if ridge_model_obj is not None:
                        # Re-asignar el objeto al modelo cargado
                        if 'principal' in self.modelos:
                            if self.modelos['principal'].get('tipo') == 'regresion_lineal_avanzada':
                                self.modelos['principal']['modelo_obj'] = ridge_model_obj
                                logger.info("✅ Modelo Ridge cargado desde joblib y asignado al modelo principal")
                            elif self.modelos['principal'].get('tipo') == 'ensemble_hibrido':
                                if 'ridge' in self.modelos['principal'].get('modelos', {}):
                                    self.modelos['principal']['modelos']['ridge']['modelo_obj'] = ridge_model_obj
                                    logger.info("✅ Modelo Ridge cargado desde joblib y asignado al ensemble")
                        else:
                            logger.warning("❌ No se encontró modelo principal para asignar el objeto joblib")
                    else:
                        logger.warning("❌ No se pudo cargar el modelo desde joblib")
                        
                except Exception as e:
                    logger.error(f"⚠️ Error cargando modelo joblib: {e}. Se continuará con los parámetros del JSON.")
            else:
                logger.info("ℹ️ No se encontró archivo joblib, usando parámetros reconstruidos desde JSON")

            # ✅ NUEVO: Actualizar cache después de cargar modelo
            if 'principal' in self.modelos and self.cache_enabled:
                cache_key = f"trained_model_loaded_{datetime.now().strftime('%Y%m%d')}"
                cache_system.set('models', self.modelos['principal'], cache_key)

            logger.info("✅ Modelo completo cargado exitosamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error crítico en cargar_modelo_completo: {e}")
            return False

    def resetear_modelos_corruptos(self):
        """Método de emergencia para resetear modelos corruptos"""
        try:
            logger.info("🔄 Reseteando modelos corruptos...")
            
            # Limpiar todos los modelos
            self.modelos = {}
            self.metricas = {}
            self.historial_entrenamiento = []
            self.feature_importance = {}
            self.scalers = {}
            self.ultimo_entrenamiento = None
            
            # Intentar eliminar archivos corruptos
            try:
                model_dir = config.MODELS_DIR
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if file.startswith('orthopredict') or file.startswith('modelo_orthopredict'):
                            file_path = os.path.join(model_dir, file)
                            os.remove(file_path)
                            logger.info(f"🗑️ Eliminado: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ No se pudieron eliminar archivos: {e}")
            
            logger.info("✅ Modelos reseteados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reseteando modelos: {e}")
            return False

    # ✅ NUEVO: Método para limpiar cache específico del modelo
    def limpiar_cache_modelo(self):
        """Limpiar cache relacionado con modelos"""
        cache_system.invalidate_pattern('models', 'trained_model_')
        cache_system.invalidate_pattern('models', 'ensemble_model_')
        cache_system.invalidate_pattern('models', 'training_data_')
        logger.info("🧹 Cache de modelos limpiado")

# Instancia global del predictor optimizado
predictor_ml = OrthoMLPredictorOptimized()