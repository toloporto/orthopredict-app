# orthopredict_app/src/data_integration.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from clinical_data_simulator import clinical_simulator

logger = logging.getLogger(__name__)

class DataIntegrationSystem:
    """Sistema de integración de datos clínicos realistas"""
    
    def __init__(self):
        self.simulated_data = None
        self.data_quality_metrics = {}
    
    def initialize_clinical_dataset(self, n_patients=2000):
        """Inicializar dataset clínico realista"""
        logger.info(f"Inicializando dataset clínico con {n_patients} pacientes")
        
        try:
            self.simulated_data = clinical_simulator.generate_realistic_clinical_dataset(
                n_patients=n_patients, 
                include_outcomes=True
            )
            
            # Calcular métricas de calidad
            self._calculate_data_quality_metrics()
            
            logger.info("Dataset clínico inicializado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando dataset clínico: {e}")
            return False
    
    def get_training_data(self) -> pd.DataFrame:
        """Obtener datos para entrenamiento de modelos"""
        if self.simulated_data is None:
            self.initialize_clinical_dataset()
        
        # Seleccionar características para el modelo
        features = [
            'age', 'gender', 'initial_crowding_mm', 'initial_overjet_mm', 
            'initial_overbite_mm', 'requires_extractions', 'compliance_level',
            'complexity_score', 'malocclusion_type'
        ]
        
        target = 'actual_treatment_duration_months'
        
        if target not in self.simulated_data.columns:
            logger.error("Dataset no contiene outcomes de tratamiento")
            return pd.DataFrame()
        
        training_data = self.simulated_data[features + [target]].copy()
        
        # Preprocesar variables categóricas
        training_data = self._preprocess_categorical_features(training_data)
        
        return training_data
    
    def _preprocess_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesar características categóricas"""
        df_processed = df.copy()
        
        # Codificar género
        df_processed['gender_encoded'] = df_processed['gender'].map({'F': 0, 'M': 1})
        
        # Codificar cooperación
        compliance_map = {'poor': 0, 'medium': 1, 'good': 2}
        df_processed['compliance_encoded'] = df_processed['compliance_level'].map(compliance_map)
        
        # Codificar tipo de maloclusión (one-hot encoding simplificado)
        malocclusion_dummies = pd.get_dummies(df_processed['malocclusion_type'], prefix='malocclusion')
        df_processed = pd.concat([df_processed, malocclusion_dummies], axis=1)
        
        # Codificar extracciones
        df_processed['extractions_encoded'] = df_processed['requires_extractions'].astype(int)
        
        return df_processed
    
    def _calculate_data_quality_metrics(self):
        """Calcular métricas de calidad del dataset"""
        if self.simulated_data is None:
            return
        
        metrics = {}
        
        # Completitud
        metrics['completeness'] = 1.0 - self.simulated_data.isnull().sum().sum() / (self.simulated_data.shape[0] * self.simulated_data.shape[1])
        
        # Distribución de duraciones
        if 'actual_treatment_duration_months' in self.simulated_data.columns:
            duration_stats = self.simulated_data['actual_treatment_duration_months'].describe()
            metrics['duration_stats'] = {
                'mean': duration_stats['mean'],
                'std': duration_stats['std'],
                'min': duration_stats['min'],
                'max': duration_stats['max']
            }
        
        # Variabilidad de características
        metrics['feature_variability'] = {
            'crowding_std': self.simulated_data['initial_crowding_mm'].std(),
            'age_range': self.simulated_data['age'].max() - self.simulated_data['age'].min(),
            'complexity_range': self.simulated_data['complexity_score'].max() - self.simulated_data['complexity_score'].min()
        }
        
        self.data_quality_metrics = metrics
        logger.info(f"Métricas de calidad calculadas: {metrics}")
    
    def get_data_quality_report(self) -> Dict:
        """Obtener reporte de calidad de datos"""
        return self.data_quality_metrics
    
    def generate_synthetic_patient(self, patient_profile: Optional[Dict] = None) -> Dict:
        """Generar paciente sintético individual"""
        if patient_profile:
            # Usar perfil personalizado si se proporciona (usa el método de esta misma clase)
            return self._generate_custom_patient(patient_profile)
        else:
            # Generar paciente aleatorio
            return clinical_simulator._generate_single_patient(
                patient_id=np.random.randint(10000, 99999), 
                include_outcomes=True
            )
    
    def _generate_custom_patient(self, patient_profile: Dict) -> Dict:
        """Generar paciente con características personalizadas"""
        base_patient = clinical_simulator._generate_single_patient(
            patient_id=patient_profile.get('patient_id', np.random.randint(10000, 99999)),
            include_outcomes=True
        )
        
        # Sobrescribir con características personalizadas
        for key, value in patient_profile.items():
            if key in base_patient:
                base_patient[key] = value
        
        # Recalcular métricas derivadas
        base_patient['complexity_score'] = clinical_simulator._calculate_complexity_score(
            base_patient['initial_crowding_mm'],
            base_patient['initial_overjet_mm'],
            base_patient['initial_overbite_mm'],
            base_patient['requires_extractions'],
            base_patient['compliance_level']
        )
        
        # Recalcular duración si hay cambios
        if any(key in patient_profile for key in ['initial_crowding_mm', 'initial_overjet_mm', 
                                                'initial_overbite_mm', 'requires_extractions', 
                                                'compliance_level']):
            base_patient['actual_treatment_duration_months'] = clinical_simulator._calculate_actual_duration(base_patient)
            base_patient['treatment_success'] = clinical_simulator._assess_treatment_success(
                base_patient['actual_treatment_duration_months'],
                base_patient['complexity_score']
            )
        
        return base_patient
        
    def validate_patient_data(self, patient_data: Dict) -> tuple[bool, List[str]]:
        """Validar datos de paciente contra distribuciones realistas"""
        errors = []
        
        # Validar rangos clínicos
        if 'initial_crowding_mm' in patient_data:
            crowding = patient_data['initial_crowding_mm']
            if not (2.0 <= crowding <= 10.0):
                errors.append(f"Apiñamiento fuera de rango realista: {crowding}mm")
        
        if 'age' in patient_data:
            age = patient_data['age']
            if not (12 <= age <= 50):
                errors.append(f"Edad fuera de rango típico de ortodoncia: {age} años")
        
        if 'initial_overjet_mm' in patient_data:
            overjet = patient_data['initial_overjet_mm']
            if not (-2.0 <= overjet <= 8.0):
                errors.append(f"Sobresalte fuera de rango realista: {overjet}mm")
        
        return len(errors) == 0, errors

# Instancia global
data_integration = DataIntegrationSystem()