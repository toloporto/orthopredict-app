# orthopredict_app/tests/test_utils.py
import pytest
import pandas as pd
from utils import OrthoUtils

class TestUtils:
    
    def test_patient_data_validation_valid(self):
        """Test validación de datos de paciente válidos"""
        datos_validos = {
            'edad': 25,
            'apiñamiento_mm': 6.0,
            'sexo': 'M',
            'sobremordida_mm': 2.5,
            'sobresalte_mm': 3.0
        }
        
        es_valido, errores = OrthoUtils.validate_patient_data(datos_validos)
        assert es_valido == True
        assert len(errores) == 0
    
    def test_patient_data_validation_invalid(self):
        """Test validación de datos de paciente inválidos"""
        datos_invalidos = {
            'edad': 50,  # Edad fuera de rango
            'apiñamiento_mm': 10.0,  # Apiñamiento fuera de rango
            'sexo': 'X'  # Sexo inválido
        }
        
        es_valido, errores = OrthoUtils.validate_patient_data(datos_invalidos)
        assert es_valido == False
        assert len(errores) > 0
    
    def test_complexity_score_calculation(self):
        """Test cálculo de score de complejidad"""
        datos_paciente = {
            'edad': 25,
            'apiñamiento_mm': 6.0,
            'sobremordida_mm': 2.5,
            'sobresalte_mm': 3.0
        }
        
        score = OrthoUtils.calculate_complexity_score(datos_paciente)
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
    
    def test_safe_json_serialization(self):
        """Test serialización segura a JSON"""
        # Test con numpy types
        import numpy as np
        datos = {
            'entero': np.int64(42),
            'flotante': np.float64(3.14),
            'array': np.array([1, 2, 3])
        }
        
        serializado = OrthoUtils.safe_json_serialize(datos)
        assert isinstance(serializado, dict)
        assert isinstance(serializado['entero'], int)
        assert isinstance(serializado['flotante'], float)
        assert isinstance(serializado['array'], list)
    
    def test_export_to_excel(self, tmp_path):
        """Test exportación a Excel"""
        datos = [
            {'nombre': 'Paciente 1', 'edad': 25, 'score': 85.5},
            {'nombre': 'Paciente 2', 'edad': 30, 'score': 92.0}
        ]
        
        file_path = tmp_path / "test_export.xlsx"
        resultado = OrthoUtils.export_to_excel(datos, str(file_path))
        
        assert resultado == True
        assert file_path.exists()
        
        # Verificar que se puede leer el archivo
        df_leido = pd.read_excel(file_path)
        assert len(df_leido) == 2