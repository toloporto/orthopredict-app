# orthopredict_app/src/clinical_data_simulator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from scipy import stats

logger = logging.getLogger(__name__)

class ClinicalDataSimulator:
    """Simulador de datos clínicos realistas basado en literatura científica"""
    
    def __init__(self):
        # Parámetros basados en estudios clínicos reales
        self.clinical_parameters = {
            # Distribución de apiñamiento en población (mm)
            'crowding_dist': {'mean': 5.8, 'std': 1.2, 'min': 2.0, 'max': 10.0},
            
            # Factores que afectan duración del tratamiento (meses)
            'base_duration_factors': {
                'age_impact': 0.15,  # meses por año sobre 18
                'crowding_impact': 1.8,  # meses por mm sobre 4mm
                'overjet_impact': 0.3,   # meses por mm de desviación del ideal (2-3mm)
                'overbite_impact': 0.4,  # meses por mm de desviación del ideal (2-3mm)
                'extraction_impact': 4.0,  # meses adicionales por extracciones
                'compliance_impact': 6.0,  # rango de impacto por cooperación
            },
            
            # Prevalencia de características clínicas (%)
            'prevalence': {
                'extraction_cases': 0.25,  # 25% de casos requieren extracciones
                'poor_compliance': 0.15,   # 15% con baja cooperación
                'deep_bite': 0.20,         # 20% con sobremordida profunda
                'open_bite': 0.08,         # 8% con mordida abierta
                'cross_bite': 0.12,        # 12% con mordida cruzada
            }
        }
    
    def generate_realistic_clinical_dataset(self, n_patients=1000, include_outcomes=True):
        """Generar dataset clínico realista con seguimiento"""
        logger.info(f"Generando dataset clínico realista de {n_patients} pacientes")
        
        patients_data = []
        
        for i in range(n_patients):
            patient = self._generate_single_patient(i, include_outcomes)
            patients_data.append(patient)
        
        df = pd.DataFrame(patients_data)
        
        # Validar distribuciones
        self._validate_distributions(df)
        
        return df
    
    def _generate_single_patient(self, patient_id, include_outcomes=True):
        """Generar datos de un solo paciente realista"""
        # Datos demográficos
        age = self._sample_age()
        gender = self._sample_gender()
        
        # Medidas clínicas iniciales
        crowding = self._sample_crowding()
        overjet = self._sample_overjet()
        overbite = self._sample_overbite()
        
        # Características clínicas adicionales
        extraction_case = self._sample_extraction_case()
        compliance_level = self._sample_compliance()
        malocclusion_type = self._classify_malocclusion(crowding, overjet, overbite)
        
        # Calcular complejidad
        complexity_score = self._calculate_complexity_score(
            crowding, overjet, overbite, extraction_case, compliance_level
        )
        
        patient_data = {
            'patient_id': f"PAT_{patient_id:04d}",
            'age': age,
            'gender': gender,
            'initial_crowding_mm': crowding,
            'initial_overjet_mm': overjet,
            'initial_overbite_mm': overbite,
            'requires_extractions': extraction_case,
            'compliance_level': compliance_level,
            'malocclusion_type': malocclusion_type,
            'complexity_score': complexity_score,
            'treatment_start_date': self._sample_start_date(),
        }
        
        # Si incluye outcomes, generar duración real del tratamiento
        if include_outcomes:
            actual_duration = self._calculate_actual_duration(patient_data)
            patient_data['actual_treatment_duration_months'] = actual_duration
            patient_data['treatment_success'] = self._assess_treatment_success(actual_duration, complexity_score)
        
        return patient_data
    
    def _sample_age(self):
        """Muestrear edad realista (12-50 años, concentrado en adolescentes)"""
        # Distribución bimodal: pico en adolescentes y adultos jóvenes
        if np.random.random() < 0.7:  # 70% adolescentes
            return int(np.random.normal(16, 2))
        else:  # 30% adultos
            return int(np.random.normal(28, 8))
    
    def _sample_gender(self):
        """Muestrear género (ligeramente más mujeres en ortodoncia)"""
        return np.random.choice(['F', 'M'], p=[0.55, 0.45])
    
    def _sample_crowding(self):
        """Muestrear apiñamiento basado en distribuciones reales"""
        crowding = np.random.normal(
            self.clinical_parameters['crowding_dist']['mean'],
            self.clinical_parameters['crowding_dist']['std']
        )
        return np.clip(round(crowding, 1), 2.0, 10.0)
    
    def _sample_overjet(self):
        """Muestrear sobresalte (rango normal 2-3mm)"""
        # 70% normal, 15% aumentado, 15% reducido
        choice = np.random.random()
        if choice < 0.7:
            return round(np.random.normal(2.5, 0.3), 1)  # Normal
        elif choice < 0.85:
            return round(np.random.uniform(4.0, 8.0), 1)  # Aumentado
        else:
            return round(np.random.uniform(-2.0, 1.5), 1)  # Reducido/negativo
    
    def _sample_overbite(self):
        """Muestrear sobremordida (rango normal 2-3mm)"""
        # 65% normal, 20% profunda, 15% abierta
        choice = np.random.random()
        if choice < 0.65:
            return round(np.random.normal(2.5, 0.4), 1)  # Normal
        elif choice < 0.85:
            return round(np.random.uniform(4.0, 8.0), 1)  # Profunda
        else:
            return round(np.random.uniform(-2.0, 1.0), 1)  # Abierta
    
    def _sample_extraction_case(self):
        """Determinar si requiere extracciones"""
        return np.random.random() < self.clinical_parameters['prevalence']['extraction_cases']
    
    def _sample_compliance(self):
        """Muestrear nivel de cooperación del paciente"""
        # Distribución: 60% buena, 25% media, 15% pobre
        choice = np.random.random()
        if choice < 0.6:
            return 'good'
        elif choice < 0.85:
            return 'medium'
        else:
            return 'poor'
    
    def _classify_malocclusion(self, crowding, overjet, overbite):
        """Clasificar tipo de maloclusión basado en medidas"""
        if crowding >= 7.0:
            return 'Severe Crowding'
        elif overjet >= 6.0:
            return 'Class II'
        elif overjet <= 0.0:
            return 'Class III'
        elif overbite >= 6.0:
            return 'Deep Bite'
        elif overbite <= 0.0:
            return 'Open Bite'
        elif crowding >= 5.0:
            return 'Moderate Crowding'
        else:
            return 'Mild Malocclusion'
    
    def _calculate_complexity_score(self, crowding, overjet, overbite, extraction_case, compliance):
        """Calcular score de complejidad 0-100"""
        score = 0
        
        # Apiñamiento (0-35 puntos)
        score += min(35, (crowding - 2) * 5)
        
        # Sobresalte anormal (0-25 puntos)
        overjet_dev = abs(overjet - 2.5)
        score += min(25, overjet_dev * 5)
        
        # Sobremordida anormal (0-20 puntos)
        overbite_dev = abs(overbite - 2.5)
        score += min(20, overbite_dev * 4)
        
        # Extracciones (10 puntos)
        if extraction_case:
            score += 10
        
        # Cooperación (0-10 puntos)
        if compliance == 'poor':
            score += 10
        elif compliance == 'medium':
            score += 5
        
        return min(100, int(score))
    
    def _calculate_actual_duration(self, patient_data):
        """Calcular duración realista del tratamiento basada en factores clínicos"""
        base_duration = 18  # meses base para caso simple
        
        # Impacto por edad
        age_impact = max(0, patient_data['age'] - 18) * self.clinical_parameters['base_duration_factors']['age_impact']
        
        # Impacto por apiñamiento
        crowding_impact = max(0, patient_data['initial_crowding_mm'] - 4) * self.clinical_parameters['base_duration_factors']['crowding_impact']
        
        # Impacto por sobresalte anormal
        overjet_impact = abs(patient_data['initial_overjet_mm'] - 2.5) * self.clinical_parameters['base_duration_factors']['overjet_impact']
        
        # Impacto por sobremordida anormal
        overbite_impact = abs(patient_data['initial_overbite_mm'] - 2.5) * self.clinical_parameters['base_duration_factors']['overbite_impact']
        
        # Impacto por extracciones
        extraction_impact = self.clinical_parameters['base_duration_factors']['extraction_impact'] if patient_data['requires_extractions'] else 0
        
        # Impacto por cooperación
        compliance_multiplier = {
            'good': 1.0,
            'medium': 1.3,
            'poor': 1.7
        }[patient_data['compliance_level']]
        
        # Duración calculada
        calculated_duration = (base_duration + age_impact + crowding_impact + 
                             overjet_impact + overbite_impact + extraction_impact) * compliance_multiplier
        
        # Añadir variabilidad natural
        variability = np.random.normal(0, 2.0)  # ±2 meses de variabilidad
        final_duration = max(12, min(36, calculated_duration + variability))
        
        return round(final_duration, 1)
    
    def _assess_treatment_success(self, actual_duration, complexity_score):
        """Evaluar éxito del tratamiento basado en duración y complejidad"""
        # Duración esperada basada en complejidad
        expected_duration = 12 + (complexity_score / 100) * 24
        
        # Considerar exitoso si está dentro del 20% de la duración esperada
        return abs(actual_duration - expected_duration) / expected_duration <= 0.2
    
    def _sample_start_date(self):
        """Generar fecha de inicio realista (últimos 5 años)"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        random_date = start_date + timedelta(days=np.random.randint(0, 5*365))
        return random_date.strftime('%Y-%m-%d')
    
    def _validate_distributions(self, df):
        """Validar que las distribuciones generadas sean realistas"""
        logger.info("Validando distribuciones del dataset...")
        
        # Verificar distribución de edades
        age_stats = df['age'].describe()
        logger.info(f"Distribución de edades: {age_stats}")
        
        # Verificar distribución de apiñamiento
        crowding_stats = df['initial_crowding_mm'].describe()
        logger.info(f"Distribución de apiñamiento: {crowding_stats}")
        
        # Verificar prevalencia de extracciones
        extraction_rate = df['requires_extractions'].mean()
        logger.info(f"Tasa de extracciones: {extraction_rate:.1%}")
        
        # Verificar distribución de duraciones
        if 'actual_treatment_duration_months' in df.columns:
            duration_stats = df['actual_treatment_duration_months'].describe()
            logger.info(f"Distribución de duraciones: {duration_stats}")

# Instancia global
clinical_simulator = ClinicalDataSimulator()