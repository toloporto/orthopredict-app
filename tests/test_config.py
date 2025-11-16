# orthopredict_app/tests/test_config.py
import pytest
import os
from config import OrthoConfig

class TestConfig:
    
    def test_config_initialization(self):
        """Test inicialización de configuración"""
        config = OrthoConfig()
        
        assert config.APP_NAME == "OrthoPredict Pro ML"
        assert config.APP_VERSION == "5.2"
        assert isinstance(config.DEBUG, bool)
    
    def test_path_creation(self, tmp_path):
        """Test creación de directorios"""
        # Usar temporary directory para pruebas
        test_data_dir = tmp_path / "test_data"
        
        # Mockear la variable de entorno
        os.environ['ORTHOPREDICT_DATA_DIR'] = str(test_data_dir)
        
        config = OrthoConfig()
        
        assert config.DATA_DIR == str(test_data_dir)
        assert os.path.exists(config.DATA_DIR)
    
    def test_environment_variables(self):
        """Test variables de entorno"""
        # Establecer variables de entorno de prueba
        os.environ['ORTHOPREDICT_DEBUG'] = 'true'
        os.environ['ORTHOPREDICT_ML_TRAINING_SAMPLES'] = '1000'
        
        config = OrthoConfig()
        
        assert config.DEBUG == True
        assert config.ML_TRAINING_SAMPLES == 1000
        
        # Limpiar
        del os.environ['ORTHOPREDICT_DEBUG']
        del os.environ['ORTHOPREDICT_ML_TRAINING_SAMPLES']
    
    def test_config_to_dict(self):
        """Test conversión a diccionario"""
        config = OrthoConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'app_name' in config_dict
        assert 'app_version' in config_dict
        assert 'debug' in config_dict