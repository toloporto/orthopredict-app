# orthopredict_app/tests/conftest.py
import pytest
import sys
import os
from pathlib import Path

# Añadir src al path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

@pytest.fixture
def sample_patient_data():
    return {
        'nombre': 'Paciente Test',
        'edad': 25,
        'sexo': 'M',
        'apiñamiento_mm': 6.0,
        'sobremordida_mm': 2.5,
        'sobresalte_mm': 3.0,
        'observaciones': 'Paciente de prueba'
    }

@pytest.fixture
def sample_training_data():
    return [
        {
            'edad': 22,
            'sexo': 'F',
            'apiñamiento_mm': 5.0,
            'sobremordida_mm': 2.0,
            'sobresalte_mm': 2.5,
            'duracion_real_meses': 16.0
        },
        {
            'edad': 28,
            'sexo': 'M',
            'apiñamiento_mm': 7.0,
            'sobremordida_mm': 3.0,
            'sobresalte_mm': 3.5,
            'duracion_real_meses': 22.0
        }
    ]