# orthopredict_app/src/utils.py
import pandas as pd
import numpy as np
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import hashlib
import joblib
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)

class OrthoUtils:
    """Utilidades compartidas para OrthoPredict"""
    
    @staticmethod
    def safe_json_serialize(obj: Any) -> Any:
        """Serializar objeto a JSON de forma segura"""
        # Casos recursivos primero para manejar estructuras anidadas
        if isinstance(obj, dict):
            return {k: OrthoUtils.safe_json_serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [OrthoUtils.safe_json_serialize(v) for v in obj]
        # Tipos de Numpy
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    @staticmethod
    def validate_patient_data(patient_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validar datos del paciente"""
        errors = []
        required_fields = ['edad', 'apiñamiento_mm', 'sexo']
        
        # Verificar campos requeridos
        for field in required_fields:
            if field not in patient_data or patient_data[field] is None:
                errors.append(f"Campo requerido faltante: {field}")
        
        # Validar rangos
        if 'edad' in patient_data:
            edad = patient_data['edad']
            if not (18 <= edad <= 35):
                errors.append("Edad debe estar entre 18 y 35 años")
        
        if 'apiñamiento_mm' in patient_data:
            apiñamiento = patient_data['apiñamiento_mm']
            if not (4.0 <= apiñamiento <= 8.0):
                errors.append("Apiñamiento debe estar entre 4.0 y 8.0 mm")
        
        if 'sobremordida_mm' in patient_data:
            sobremordida = patient_data['sobremordida_mm']
            if not (1.0 <= sobremordida <= 5.0):
                errors.append("Sobremordida debe estar entre 1.0 y 5.0 mm")
        
        if 'sobresalte_mm' in patient_data:
            sobresalte = patient_data['sobresalte_mm']
            if not (1.0 <= sobresalte <= 5.0):
                errors.append("Sobresalte debe estar entre 1.0 y 5.0 mm")
        
        # Validar sexo
        if 'sexo' in patient_data:
            sexo = patient_data['sexo']
            if sexo not in ['M', 'F', 'Masculino', 'Femenino']:
                errors.append("Sexo debe ser 'M', 'F', 'Masculino' o 'Femenino'")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def calculate_complexity_score(patient_data: Dict[str, Any]) -> float:
        """Calcular score de complejidad del caso (0-100)"""
        score = 0
        
        # Apiñamiento (40% del score)
        apiñamiento = patient_data.get('apiñamiento_mm', 6)
        apiñamiento_score = min(40, (apiñamiento - 4) * 10)  # 4mm=0, 8mm=40
        score += apiñamiento_score
        
        # Edad (20% del score)
        edad = patient_data.get('edad', 25)
        edad_score = min(20, max(0, (edad - 18) * 1.18))  # 18 años=0, 35 años=20
        score += edad_score
        
        # Sobremordida (20% del score)
        sobremordida = patient_data.get('sobremordida_mm', 2.5)
        sobremordida_score = min(20, abs(sobremordida - 2.5) * 8)  # 2.5=0, 1.0 o 5.0=20
        score += sobremordida_score
        
        # Sobresalte (20% del score)
        sobresalte = patient_data.get('sobresalte_mm', 3.0)
        sobresalte_score = min(20, abs(sobresalte - 3.0) * 6.67)  # 3.0=0, 1.0 o 5.0=20
        score += sobresalte_score
        
        return round(score, 1)
    
    @staticmethod
    def export_to_excel(data: Union[Dict, List, pd.DataFrame], 
                       file_path: str, 
                       sheet_name: str = 'Data') -> bool:
        """Exportar datos a Excel usando openpyxl"""
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                logger.error(f"Tipo de datos no soportado para exportación Excel: {type(data)}")
                return False
            
            # Crear directorio si no existe
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Exportar a Excel
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-ajustar columnas
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            logger.info(f"Datos exportados exitosamente a: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando a Excel: {e}")
            return False
    
    @staticmethod
    def save_model_joblib(model: Any, file_path: str) -> bool:
        """Guardar modelo usando joblib (más eficiente que pickle)"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, file_path, compress=3)  # Compresión nivel 3
            logger.info(f"Modelo guardado con joblib: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error guardando modelo con joblib: {e}")
            return False
    
    @staticmethod
    def load_model_joblib(file_path: str) -> Optional[Any]:
        """Cargar modelo usando joblib"""
        try:
            if not Path(file_path).exists():
                logger.warning(f"Archivo de modelo no encontrado: {file_path}")
                return None
            
            model = joblib.load(file_path)
            logger.info(f"Modelo cargado con joblib: {file_path}")
            return model
        except Exception as e:
            logger.error(f"Error cargando modelo con joblib: {e}")
            return None
    
    @staticmethod
    def generate_patient_id() -> str:
        """Generar ID único para paciente"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        random_component = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"ORTHO_{timestamp}_{random_component}"
    
    @staticmethod
    def format_duration_months(months: float) -> str:
        """Formatear duración en meses a formato legible"""
        if months < 12:
            return f"{months:.1f} meses"
        else:
            years = months / 12
            if years.is_integer():
                return f"{int(years)} año{'s' if years > 1 else ''}"
            else:
                return f"{years:.1f} años"
    
    @staticmethod
    def calculate_confidence_interval(prediction: float, error: float, confidence: float = 0.95) -> tuple[float, float]:
        """Calcular intervalo de confianza"""
        from scipy import stats
        z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
        margin = error * z_score
        return max(0, prediction - margin), prediction + margin

# Instancia global de utilidades
utils = OrthoUtils()