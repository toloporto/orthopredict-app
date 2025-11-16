# orthopredict_app/src/config.py
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging
from datetime import datetime  # ✅ AÑADIR ESTA LÍNEA

# Cargar variables de entorno
load_dotenv(encoding='utf-8')

class OrthoConfig:
    """Sistema centralizado de configuración para OrthoPredict"""
    
    def __init__(self):
        # Configuración de la aplicación
        self.APP_NAME = "OrthoPredict Pro ML"
        self.APP_VERSION = "5.2"
        self.DEBUG = self._get_bool('DEBUG', False)
        
        # Configuración de rutas
        self.DATA_DIR = self._get_path('DATA_DIR', 'data')
        self.MODELS_DIR = self._get_path('MODELS_DIR', 'models')
        self.LOGS_DIR = self._get_path('LOGS_DIR', 'logs')
        self.BACKUP_DIR = self._get_path('BACKUP_DIR', 'backups')
        
        # Configuración ML
        self.ML_CACHE_SIZE = self._get_int('ML_CACHE_SIZE', 100)
        self.ML_PREDICTION_TIMEOUT = self._get_int('ML_PREDICTION_TIMEOUT', 30)
        self.ML_TRAINING_SAMPLES = self._get_int('ML_TRAINING_SAMPLES', 2000)
        
        # Añadir configuración para modelos
        self.ML_HYPERPARAM_OPTIMIZATION = self._get_bool('ML_HYPERPARAM_OPTIMIZATION', True)
        self.ML_EARLY_STOPPING = self._get_bool('ML_EARLY_STOPPING', True)
        
        # Configuración de monitoreo
        self.MONITORING_ENABLED = self._get_bool('MONITORING_ENABLED', True)
        self.MONITORING_INTERVAL = self._get_int('MONITORING_INTERVAL', 3600)  # 1 hora
        self.ALERT_THRESHOLD = self._get_float('ALERT_THRESHOLD', 0.7)
        
        # Configuración de cache
        self.CACHE_ENABLED = self._get_bool('CACHE_ENABLED', True)
        self.CACHE_TTL = self._get_int('CACHE_TTL', 3600)  # 1 hora
        self.CACHE_MAX_SIZE = self._get_int('CACHE_MAX_SIZE', 1000)
        
        # Configuración de reportes
        self.REPORT_FORMATS = self._get_list('REPORT_FORMATS', ['pdf', 'excel', 'csv'])
        self.REPORT_RETENTION_DAYS = self._get_int('REPORT_RETENTION_DAYS', 30)
        
        # Configuración de seguridad
        self.SESSION_TIMEOUT = self._get_int('SESSION_TIMEOUT', 3600)  # 1 hora
        self.MAX_LOGIN_ATTEMPTS = self._get_int('MAX_LOGIN_ATTEMPTS', 5)
        
        # ✅ NUEVO: Configuración de datos clínicos
        self.CLINICAL_DATA_ENABLED = self._get_bool('CLINICAL_DATA_ENABLED', True)
        self.CLINICAL_DATASET_SIZE = self._get_int('CLINICAL_DATASET_SIZE', 2000)
        self.ANALYTICS_ENABLED = self._get_bool('ANALYTICS_ENABLED', True)
        self.CLINICAL_REPORT_RETENTION = self._get_int('CLINICAL_REPORT_RETENTION', 90)  # días
        self.AUTO_CLINICAL_ANALYSIS = self._get_bool('AUTO_CLINICAL_ANALYSIS', True)
        
        # Crear directorios necesarios
        self._create_directories()
        
        # Configurar logging
        self._setup_logging()
    
    def _get_env(self, key: str, default: Any = None) -> Optional[str]:
        """Obtener variable de entorno"""
        return os.getenv(f'ORTHOPREDICT_{key}', default)
    
    def _get_int(self, key: str, default: int) -> int:
        """Obtener variable de entorno como entero"""
        try:
            return int(self._get_env(key, default))
        except (ValueError, TypeError):
            return default
    
    def _get_float(self, key: str, default: float) -> float:
        """Obtener variable de entorno como float"""
        try:
            return float(self._get_env(key, default))
        except (ValueError, TypeError):
            return default
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """Obtener variable de entorno como booleano"""
        value = self._get_env(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _get_list(self, key: str, default: list) -> list:
        """Obtener variable de entorno como lista"""
        value = self._get_env(key, '')
        if not value:
            return default
        return [item.strip() for item in value.split(',')]
    
    def _get_path(self, key: str, default: str) -> str:
        """Obtener ruta y expandir variables de usuario"""
        path = self._get_env(key, default)
        return os.path.expanduser(path)
    
    def _create_directories(self):
        """Crear directorios necesarios"""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            self.BACKUP_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_logging(self):
        """Configurar sistema de logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configurar root logger
        logging.basicConfig(
            level=logging.DEBUG if self.DEBUG else logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(self.LOGS_DIR, 'orthopredict.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # Reducir verbosidad de librerías externas
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('plotly').setLevel(logging.WARNING)
    
    def get_database_path(self) -> str:
        """Obtener ruta de la base de datos"""
        return os.path.join(self.DATA_DIR, 'pacientes_db.json')
    
    def get_model_path(self, model_name: str = 'orthopredict') -> str:
        """Obtener ruta para modelos"""
        return os.path.join(self.MODELS_DIR, f'{model_name}.json')
    
    def get_backup_path(self, backup_type: str = 'manual') -> str:
        """Obtener ruta para backups"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.BACKUP_DIR, f'{backup_type}_{timestamp}.zip')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario (sin información sensible)"""
        return {
            'app_name': self.APP_NAME,
            'app_version': self.APP_VERSION,
            'debug': self.DEBUG,
            'data_dir': self.DATA_DIR,
            'models_dir': self.MODELS_DIR,
            'ml_cache_size': self.ML_CACHE_SIZE,
            'monitoring_enabled': self.MONITORING_ENABLED,
            'cache_enabled': self.CACHE_ENABLED,
            'clinical_data_enabled': self.CLINICAL_DATA_ENABLED,
            'report_formats': self.REPORT_FORMATS
        }

# Instancia global de configuración
config = OrthoConfig()