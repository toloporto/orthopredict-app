# orthopredict_app/tests/test_ml_models.py
import pytest
import pandas as pd
import numpy as np
from ml_models import OrthoMLPredictorOptimized

class TestMLModels:
    
    def test_predictor_initialization(self):
        """Test que el predictor se inicializa correctamente"""
        predictor = OrthoMLPredictorOptimized()
        assert predictor.modelos == {}
        assert predictor.metricas == {}
        assert predictor.cache_enabled == True
    
    def test_prediction_with_valid_data(self, sample_patient_data):
        """Test predicción con datos válidos"""
        predictor = OrthoMLPredictorOptimized()
        resultado = predictor.predecir_duracion(sample_patient_data)
        
        # Verificar estructura de respuesta
        assert 'prediccion' in resultado
        assert 'intervalo_min' in resultado
        assert 'intervalo_max' in resultado
        assert 'modelo_usado' in resultado
        assert 'confianza' in resultado
        
        # Verificar rangos válidos
        assert 12 <= resultado['prediccion'] <= 36
        assert resultado['intervalo_min'] <= resultado['prediccion']
        assert resultado['intervalo_max'] >= resultado['prediccion']
    
    def test_prediction_with_invalid_data(self):
        """Test predicción con datos inválidos"""
        predictor = OrthoMLPredictorOptimized()
        datos_invalidos = {'edad': 50, 'apiñamiento_mm': 15}  # Datos fuera de rango
        
        resultado = predictor.predecir_duracion(datos_invalidos)
        
        # Debería retornar predicción por defecto
        assert 'prediccion' in resultado
        assert 'modelo_usado' in resultado
    
    def test_training_data_generation(self):
        """Test generación de datos de entrenamiento"""
        predictor = OrthoMLPredictorOptimized()
        datos = predictor.generar_datos_entrenamiento_avanzado(100)
        
        assert len(datos) == 100
        assert 'edad' in datos.columns
        assert 'apiñamiento_mm' in datos.columns
        assert 'duracion_real_meses' in datos.columns
        
        # Verificar rangos de datos generados
        assert datos['edad'].between(18, 35).all()
        assert datos['apiñamiento_mm'].between(4.0, 8.0).all()
    
    def test_model_training(self, sample_training_data):
        """Test entrenamiento del modelo"""
        predictor = OrthoMLPredictorOptimized()
        df_entrenamiento = pd.DataFrame(sample_training_data)
        
        modelo = predictor.entrenar_modelo_avanzado(df_entrenamiento)
        
        assert modelo is not None
        assert 'metricas' in modelo
        assert 'tipo' in modelo
        assert predictor.ultimo_entrenamiento is not None
    
    def test_feature_importance_calculation(self):
        """Test cálculo de importancia de features"""
        predictor = OrthoMLPredictorOptimized()
        
        # Generar datos de prueba
        datos = predictor.generar_datos_entrenamiento_avanzado(50)
        predictor.entrenar_modelo_avanzado(datos)
        
        assert predictor.feature_importance is not None
        assert isinstance(predictor.feature_importance, dict)
    
    def test_cache_functionality(self, sample_patient_data):
        """Test funcionalidad de cache"""
        predictor = OrthoMLPredictorOptimized()
        
        # Primera predicción (debe calcular)
        resultado1 = predictor.predecir_duracion(sample_patient_data)
        
        # Segunda predicción con mismos datos (debe usar cache)
        resultado2 = predictor.predecir_duracion(sample_patient_data)
        
        # Las predicciones deben ser consistentes
        assert resultado1['prediccion'] == resultado2['prediccion']