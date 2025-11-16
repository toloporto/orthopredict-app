# orthopredict_app/src/clinical_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ClinicalAnalyzer:
    """Analizador clínico para interpretar resultados del modelo"""
    
    def __init__(self):
        self.clinical_guidelines = {
            'duration_benchmarks': {
                'simple_cases': (12, 18),
                'moderate_cases': (18, 24),
                'complex_cases': (24, 36)
            },
            'complexity_thresholds': {
                'low': 30,
                'moderate': 50,
                'high': 70
            },
            'risk_factors': {
                'high_crowding': 6.0,  # mm
                'deep_bite': 5.0,      # mm
                'large_overjet': 6.0,  # mm
                'poor_compliance': 'poor'
            }
        }
    
    def analyze_prediction(self, patient_data: Dict, prediction_result: Dict) -> Dict:
        """Analizar predicción desde perspectiva clínica"""
        analysis = {
            'clinical_interpretation': {},
            'risk_factors': [],
            'treatment_recommendations': [],
            'expected_challenges': []
        }
        
        # Interpretar duración predicha
        duration = prediction_result.get('prediccion', 18)
        analysis['clinical_interpretation']['duration_category'] = self._classify_duration(duration)
        
        # Identificar factores de riesgo
        analysis['risk_factors'] = self._identify_risk_factors(patient_data)
        
        # Generar recomendaciones
        analysis['treatment_recommendations'] = self._generate_recommendations(patient_data, duration)
        
        # Identificar desafíos esperados
        analysis['expected_challenges'] = self._identify_challenges(patient_data)
        
        # Calcular confianza clínica
        analysis['clinical_confidence'] = self._calculate_clinical_confidence(patient_data, prediction_result)
        
        return analysis
    
    def _classify_duration(self, duration: float) -> str:
        """Clasificar duración según benchmarks clínicos"""
        if duration <= 18:
            return "Corta (Caso Simple)"
        elif duration <= 24:
            return "Moderada (Caso de Complejidad Media)"
        else:
            return "Extendida (Caso Complejo)"
    
    def _identify_risk_factors(self, patient_data: Dict) -> List[str]:
        """Identificar factores de riesgo clínicos"""
        risk_factors = []
        
        if patient_data.get('apiñamiento_mm', 0) >= self.clinical_guidelines['risk_factors']['high_crowding']:
            risk_factors.append("Apiñamiento severo")
        
        if patient_data.get('sobremordida_mm', 0) >= self.clinical_guidelines['risk_factors']['deep_bite']:
            risk_factors.append("Sobremordida profunda")
        
        if patient_data.get('sobresalte_mm', 0) >= self.clinical_guidelines['risk_factors']['large_overjet']:
            risk_factors.append("Sobresalte aumentado")
        
        # Puedes añadir más factores según tu data real
        
        return risk_factors
    
    def _generate_recommendations(self, patient_data: Dict, duration: float) -> List[str]:
        """Generar recomendaciones de tratamiento basadas en datos clínicos"""
        recommendations = []
        
        # Recomendaciones basadas en duración
        if duration > 24:
            recommendations.append("Considerar evaluación ortodóncico-quirúrgica")
            recommendations.append("Planificar citas de ajuste más frecuentes")
        elif duration > 18:
            recommendations.append("Seguimiento estándar cada 6-8 semanas")
        else:
            recommendations.append("Seguimiento convencional cada 8-10 semanas")
        
        # Recomendaciones específicas por factores
        if patient_data.get('apiñamiento_mm', 0) > 6.0:
            recommendations.append("Evaluar necesidad de extracciones para desapiñamiento")
        
        if patient_data.get('edad', 25) > 30:
            recommendations.append("Considerar movilidad dental reducida en adultos")
        
        return recommendations
    
    def _identify_challenges(self, patient_data: Dict) -> List[str]:
        """Identificar desafíos potenciales en el tratamiento"""
        challenges = []
        
        crowding = patient_data.get('apiñamiento_mm', 0)
        if crowding > 7.0:
            challenges.append("Alto potencial de recidiva por apiñamiento severo")
        
        age = patient_data.get('edad', 25)
        if age > 35:
            challenges.append("Posible remodelación ósea más lenta en adultos")
        
        return challenges
    
    def _calculate_clinical_confidence(self, patient_data: Dict, prediction_result: Dict) -> float:
        """Calcular confianza basada en consistencia clínica"""
        confidence = 80.0  # Base
        
        # Ajustar por rango de predicción
        pred_min = prediction_result.get('intervalo_min', 0)
        pred_max = prediction_result.get('intervalo_max', 0)
        pred_range = pred_max - pred_min
        
        if pred_range <= 4:
            confidence += 10
        elif pred_range >= 8:
            confidence -= 15
        
        # Ajustar por factores de riesgo conocidos
        risk_factors = self._identify_risk_factors(patient_data)
        if not risk_factors:
            confidence += 5
        elif len(risk_factors) >= 3:
            confidence -= 10
        
        return max(50, min(95, confidence))

# Instancia global
clinical_analyzer = ClinicalAnalyzer()