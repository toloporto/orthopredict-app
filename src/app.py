# orthopredict_app/src/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import json
import datetime
import os
import sys
import tempfile

from dotenv import find_dotenv, set_key
# Añadir el directorio src al path para importar nuestros módulos
sys.path.append(os.path.dirname(__file__))

# Importar nuestros sistemas optimizados
from ml_models import predictor_ml
from report_generator import PDFReportGenerator
from monitoring_system import MLMonitoringSystem
from visualization_system import viz_system  # ✅ NUEVO: Sistema de visualización
from config import config  # ✅ NUEVO: Configuración centralizada
from cache_system import cache_system  # ✅ NUEVO: Sistema de cache
from utils import utils  # ✅ NUEVO: Utilidades compartidas

from clinical_data_simulator import clinical_simulator
from data_integration import data_integration
from clinical_analyzer import clinical_analyzer
from auto_reporter import AutoReporter, ReportScheduler
from monitoring_dashboard import create_monitoring_dashboard
# Importar sistemas de autenticación y backup
try:
    from auth_system import auth_system, check_authentication, login_page, require_auth, require_role, logout
    from backup_system import backup_system
    AUTH_ENABLED = True
except ImportError:
    AUTH_ENABLED = False
    print("⚠️ Sistemas de autenticación y backup no disponibles")

# Instancias globales OPTIMIZADAS
pdf_generator = PDFReportGenerator()
sistema_monitoreo = MLMonitoringSystem()
monitoring_dashboard = create_monitoring_dashboard(sistema_monitoreo)
auto_reporter = AutoReporter(sistema_monitoreo)
report_scheduler = ReportScheduler(auto_reporter)
clinical_data = data_integration

# ✅ NUEVO: Configurar logging
import logging
logger = logging.getLogger(__name__)

# ==================== FUNCIONES AUXILIARES (MOVIDAS ARRIBA) ====================
def inicializar_base_datos():
    """Inicializar base de datos usando configuración centralizada"""
    try:
        db_path = config.get_database_path()
        
        # ✅ NUEVO: Verificar cache primero
        cache_key = "database_cache"
        cached_db = cache_system.get('patient_data', cache_key)
        if cached_db is not None:
            return cached_db
            
        with open(db_path, 'r', encoding='utf-8') as f:
            db_data = json.load(f)
            
        # ✅ NUEVO: Guardar en cache
        cache_system.set('patient_data', db_data, cache_key)
        return db_data
        
    except FileNotFoundError:
        base_datos = {
            'pacientes': [],
            'ultimo_id': 0,
            'estadisticas': {
                'total_pacientes': 0,
                'duracion_promedio': 0,
                'casos_completados': 0
            }
        }
        db_path = config.get_database_path()
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(base_datos, f, indent=2, default=utils.safe_json_serialize)  # ✅ NUEVO: Serialización segura
        return base_datos
    except Exception as e:
        logger.error(f"Error inicializando base de datos: {e}")
        # Retornar estructura básica en caso de error
        return {
            'pacientes': [],
            'ultimo_id': 0,
            'estadisticas': {'total_pacientes': 0}
        }

def guardar_paciente(nuevo_paciente):
    """Guardar paciente con validación y cache"""
    # ✅ NUEVO: Validar datos del paciente
    is_valid, errors = utils.validate_patient_data(nuevo_paciente)
    if not is_valid:
        logger.error(f"Datos de paciente inválidos: {errors}")
        raise ValueError(f"Errores en datos del paciente: {', '.join(errors)}")
    
    db = inicializar_base_datos()
    db['ultimo_id'] += 1
    nuevo_paciente['id'] = db['ultimo_id']
    nuevo_paciente['fecha_creacion'] = datetime.datetime.now().isoformat()
    
    # ✅ NUEVO: Calcular score de complejidad
    nuevo_paciente['complejidad_score'] = utils.calculate_complexity_score(nuevo_paciente)
    
    db['pacientes'].append(nuevo_paciente)
    db['estadisticas']['total_pacientes'] = len(db['pacientes'])
    
    db_path = config.get_database_path()
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, default=utils.safe_json_serialize)
    
    # ✅ NUEVO: Invalidar cache de base de datos
    cache_system.clear('patient_data')
    cache_system.invalidate_pattern('patient_data', 'stats_')
    
    logger.info(f"Paciente guardado: ORTHO-{nuevo_paciente['id']}")
    return nuevo_paciente['id']

def obtener_todos_pacientes():
    """Obtener todos los pacientes con cache"""
    try:
        # ✅ NUEVO: Verificar cache primero
        cache_key = "all_patients"
        cached_patients = cache_system.get('patient_data', cache_key)
        
        if cached_patients is not None:
            return cached_patients
            
        db = inicializar_base_datos()
        pacientes = db.get('pacientes', [])
        
        # ✅ NUEVO: Guardar en cache
        cache_system.set('patient_data', pacientes, cache_key)
        return pacientes
    except Exception as e:
        logger.error(f"Error obteniendo pacientes: {e}")
        return []

def generar_reporte_pdf(paciente_data, prediccion_data):
    """Generar reporte PDF usando el sistema optimizado"""
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            pdf_path = temp_file.name
        
        # Generar reporte
        success = pdf_generator.generar_reporte_paciente(
            paciente_data, prediccion_data, pdf_path
        )
        
        if success:
            return pdf_path
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error generando PDF: {e}")
        return None

def evaluar_modelo_periodicamente():
    """Evaluar el modelo periódicamente"""
    try:
        # Obtener datos recientes (últimos 50 pacientes)
        pacientes_recientes = obtener_todos_pacientes()[-50:]
        
        if len(pacientes_recientes) >= 10:  # Mínimo para evaluación
            df_recientes = pd.DataFrame(pacientes_recientes)
            evaluacion = sistema_monitoreo.evaluar_modelo_actual(predictor_ml, df_recientes)
            return evaluacion
        else:
            return {'error': 'Datos insuficientes para evaluación'}
    except Exception as e:
        return {'error': f'Error en evaluación periódica: {e}'}

def inicializar_sistema_monitoreo():
    """Inicializar sistema de monitoreo con datos de referencia"""
    try:
        # 1. Establecer los datos de referencia para el detector de drift
        datos_referencia = predictor_ml.generar_datos_entrenamiento_avanzado(1000)
        sistema_monitoreo.drift_detector.establecer_referencia(datos_referencia)
        
        # 2. ✅ CORRECCIÓN: Realizar la primera evaluación para activar el dashboard
        pacientes = obtener_todos_pacientes()
        if len(pacientes) >= 10:
            sistema_monitoreo.evaluar_modelo_actual(predictor_ml, pd.DataFrame(pacientes))
            
        return True
    except Exception as e:
        logger.error(f"Error inicializando monitoreo: {e}")
        return False

def obtener_datos_monitoreo():
    """Obtener datos para el dashboard de monitoreo tradicional"""
    try:
        df_recientes = pd.DataFrame(obtener_todos_pacientes()[-50:])
        if len(df_recientes) > 10:
            evaluacion_actual = sistema_monitoreo.evaluar_modelo_actual(predictor_ml, df_recientes)
        else:
            evaluacion_actual = {'error': 'Datos insuficientes'}

        return {
            'evaluacion_actual': evaluacion_actual,
            'estado_sistema': 'ACTIVO' if sistema_monitoreo.metricas_historicas else 'INACTIVO'
        }
    except Exception as e:
        logger.error(f"Error obteniendo datos de monitoreo: {e}")
        return {'estado_sistema': 'ERROR'}

# ==================== CONFIGURACIÓN INICIAL OPTIMIZADA ====================
# Limpiar modelos corruptos al inicio (solo una vez)
# ⚠️ DESACTIVADO: Esta línea borraba los modelos en cada reinicio.
# if 'models_cleaned' not in st.session_state:
#     logger.info("🔄 Limpiando modelos corruptos...")
#     predictor_ml.resetear_modelos_corruptos()
#     st.session_state.models_cleaned = True

# ✅ NUEVO: Inicializar cache system
if 'cache_initialized' not in st.session_state:
    cache_system.clear()  # Limpiar cache al inicio
    st.session_state.cache_initialized = True

# ==================== VERIFICAR AUTENTICACIÓN ====================
if AUTH_ENABLED and not check_authentication():
    login_page()
    st.stop()

# Configurar la página con información de la versión
st.set_page_config(
    page_title=f"{config.APP_NAME} v{config.APP_VERSION}",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SIDEBAR MEJORADO CON NUEVAS MÉTRICAS ====================
with st.sidebar:
    if AUTH_ENABLED:
        # Información del usuario
        st.header(f"👋 Hola, {st.session_state.name}")
        role_badge = "👑" if st.session_state.role == 'admin' else "👨‍⚕️"
        st.markdown(f"**Rol:** {role_badge} {st.session_state.role.title()}")
        st.markdown(f"**Usuario:** `{st.session_state.user}`")
        
        # Tiempo de sesión
        if st.session_state.login_time:
            session_duration = datetime.datetime.now() - st.session_state.login_time
            hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            st.markdown(f"**Sesión:** {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        st.markdown("---")
    
    # Navegación basada en roles
    st.header("🧭 Navegación")
    
    # Opciones básicas para todos los usuarios
    opciones_navegacion = [
        "🎯 Predictor ML", 
        "📊 Entrenar Modelos", 
        "🔍 Diagnóstico Modelos", 
        "👥 Gestión Pacientes", 
        "🔬 Análisis de Cohortes",
        "📈 Monitoreo ML", # ✅ CORREGIDO
        "⚙️ Sistema"
    ]
    
    # Solo administradores pueden acceder a Administración
    if AUTH_ENABLED and st.session_state.role == "admin":
        opciones_navegacion.append("👑 Administración")
    
    pagina = st.radio("Seleccionar módulo:", opciones_navegacion)
    
    st.markdown("---")
    
    # Estado del sistema OPTIMIZADO
    st.header(f"🤖 {config.APP_NAME}")
    st.markdown(f"**v{config.APP_VERSION} - Optimizado**")
    
    # ✅ NUEVO: Intentar cargar modelo con cache
    try:
        model_path = config.get_model_path()
        # Verificar cache primero
        cached_model = cache_system.get('models', 'principal_model')
        if cached_model is None:
            if predictor_ml.cargar_modelo_completo(model_path):
                # Guardar en cache
                cache_system.set('models', True, 'principal_model')
                st.success("✅ Modelo ML cargado") 
            else:
                st.info("⏳ Modelo no entrenado")
        else:
            st.success("✅ Modelo ML (en cache)")
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        st.info("⏳ Modelo no entrenado")
    
    # ✅ NUEVO: Estadísticas con cache
    st.subheader("📈 Estadísticas")
    try:
        # Usar cache para estadísticas
        stats_cache_key = f"stats_{datetime.datetime.now().strftime('%Y%m%d%H')}"
        cached_stats = cache_system.get('patient_data', stats_cache_key)
        
        if cached_stats is None:
            db = inicializar_base_datos()
            metricas = predictor_ml.obtener_metricas()
            
            stats_data = {
                'total_pacientes': db['estadisticas']['total_pacientes'],
                'mae_modelo': metricas.get('mae', 'N/A'),
                'usuarios_registrados': len(auth_system.get_all_users()) if AUTH_ENABLED else 0
            }
            
            # Guardar en cache por 1 hora
            cache_system.set('patient_data', stats_data, stats_cache_key)
        else:
            stats_data = cached_stats
        
        st.metric("Pacientes Totales", stats_data['total_pacientes'])
        
        if AUTH_ENABLED:
            st.metric("Usuarios Registrados", stats_data['usuarios_registrados'])
        
        if stats_data['mae_modelo'] != 'N/A':
            st.metric("MAE Modelo", f"{stats_data['mae_modelo']} meses")
        else:
            st.metric("MAE Modelo", "No entrenado")
            
    except Exception as e:
        logger.error(f"Error cargando estadísticas: {e}")
        st.metric("Pacientes Totales", 0)

    # ✅ NUEVO: Estado del sistema de cache
    st.subheader("⚡ Sistema Cache")
    cache_stats = cache_system.get_stats()
    st.metric("Tasa de Aciertos", f"{cache_stats['hit_rate']}%")
    st.metric("Operaciones", cache_stats['total_hits'] + cache_stats['total_misses'])
    
    st.markdown("---")
    
    # Botón de logout si la autenticación está habilitada
    if AUTH_ENABLED:
        if st.button("🚪 Cerrar Sesión", width='stretch'):
            logout()
            st.rerun()

# ==================== EVALUACIÓN PERIÓDICA Y ALERTAS ====================
try:
    # Evaluar modelo cada 24 horas
    if 'ultima_evaluacion' not in st.session_state:
        st.session_state.ultima_evaluacion = None
    
    if (st.session_state.ultima_evaluacion is None or 
        (datetime.datetime.now() - st.session_state.ultima_evaluacion).seconds > 86400):
        pacientes_recientes = obtener_todos_pacientes()[-50:]
        df_recientes = pd.DataFrame(pacientes_recientes) if pacientes_recientes else pd.DataFrame()
        evaluacion = sistema_monitoreo.evaluar_modelo_actual(predictor_ml, df_recientes) # Pasamos los datos directamente
        st.session_state.ultima_evaluacion = datetime.datetime.now()
        
        # ✅ NUEVO: Verificar reportes automáticos
        report_scheduler.check_scheduled_reports()
        
        # Mostrar alertas críticas
        if evaluacion and 'alertas_activadas' in evaluacion:
            alertas_criticas = [a for a in evaluacion['alertas_activadas'] 
                               if a.get('nivel') == 'CRITICO']
            for alerta in alertas_criticas:
                st.error(f"🚨 ALERTA CRÍTICA: {alerta.get('mensaje', '')}")
                
except Exception as e:
    logger.error(f"Error en evaluación periódica: {e}")

# ==================== PÁGINA: PREDICTOR ML OPTIMIZADO ====================
if pagina == "🎯 Predictor ML":
    st.header("🎯 Predictor con Machine Learning y Análisis Clínico")
    
    # Información del modelo con cache
    metricas = predictor_ml.obtener_metricas()
    if metricas:
        st.success(f"✅ Modelo ML activo - MAE: {metricas.get('mae', 'N/A')} meses | R²: {metricas.get('r2', 'N/A')}")
    else:
        st.warning("⚠️ Usando predictor básico - Entrena el modelo ML para mejor precisión")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.form("form_predictor_ml"):
            st.subheader("📋 Datos del Paciente")
            
            nombre = st.text_input("Nombre completo del paciente")
            edad = st.slider("Edad", 18, 35, 25)
            sexo = st.radio("Sexo", ["Femenino", "Masculino"], horizontal=True)
            
            st.subheader("🔬 Parámetros Clínicos")
            apiñamiento = st.slider("Apiñamiento inferior (mm)", 4.0, 8.0, 6.0, 0.1)
            sobremordida = st.slider("Sobremordida (mm)", 1.0, 5.0, 2.5, 0.1)
            sobresalte = st.slider("Sobresalte (mm)", 1.0, 5.0, 3.0, 0.1)
            
            observaciones = st.text_area("Observaciones clínicas")
            
            submitted = st.form_submit_button("🤖 Predecir con ML")
    
    with col2:
        st.subheader("📊 Resultados ML")
        
        if submitted:
            # ✅ NUEVO: Verificar cache de predicción
            prediction_cache_key = f"pred_{edad}_{apiñamiento}_{sobremordida}_{sobresalte}"
            cached_prediction = cache_system.get('predictions', prediction_cache_key)
            
            if cached_prediction is not None:
                resultado = cached_prediction
                st.info("⚡ Resultado obtenido desde cache")
            else:
                with st.spinner("Ejecutando modelo de machine learning..."):
                    # Preparar datos para el modelo ML
                    datos_paciente = {
                        'nombre': nombre,
                        'edad': edad,
                        'sexo': sexo,
                        'apiñamiento_mm': apiñamiento,
                        'sobremordida_mm': sobremordida,
                        'sobresalte_mm': sobresalte
                    }
                    
                    # Obtener predicción del modelo ML
                    resultado = predictor_ml.predecir_duracion(datos_paciente)
                    
                    # ✅ NUEVO: Guardar en cache
                    cache_system.set('predictions', resultado, prediction_cache_key)
            
            st.success("✅ Predicción ML completada!")
            
            # Mostrar resultados
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Duración ML", f"{resultado['prediccion']} meses")
            with col_res2:
                st.metric("Rango", f"{resultado['intervalo_min']}-{resultado['intervalo_max']} meses")
            with col_res3:
                st.metric("Confianza ML", f"{resultado['confianza']}%")
            
            # ✅ NUEVA FUNCIONALIDAD: Predicción de Éxito del Tratamiento
            st.markdown("---")
            st.subheader("🎯 Probabilidad de Éxito del Tratamiento")
            probabilidad_exito = predictor_ml.predecir_probabilidad_exito(datos_paciente)
            
            # Interpretación de la probabilidad
            if probabilidad_exito >= 0.85:
                interpretacion_exito = "Muy Alta. El plan de tratamiento tiene una alta probabilidad de cumplir los objetivos en el tiempo estimado."
            elif probabilidad_exito >= 0.70:
                interpretacion_exito = "Alta. Se espera un buen resultado con un seguimiento adecuado."
            else:
                interpretacion_exito = "Moderada. Pueden existir factores de riesgo que requieran atención especial."
            st.progress(probabilidad_exito, text=f"{probabilidad_exito:.0%} de Probabilidad de Éxito")
            st.info(f"**Interpretación:** {interpretacion_exito}")

            # Información del modelo usado
            st.info(f"**Modelo utilizado:** {resultado['modelo_usado'].replace('_', ' ').title()}")
            
            # Factores considerados
            with st.expander("🔍 Ver factores considerados por el modelo"):
                for factor in resultado['factores_considerados']:
                    st.write(f"• {factor}")
            
            # ✅ NUEVO: GRÁFICOS INTERACTIVOS CON PLOTLY
            st.subheader("📈 Análisis Visual de Predicción")
            
            # Gráfico de predicción interactivo
            fig_prediccion = viz_system.crear_grafico_prediccion(resultado, datos_paciente)
            st.plotly_chart(fig_prediccion, use_container_width=True)
            
            # Análisis de complejidad
            st.subheader("🎯 Análisis de Complejidad del Caso")
            fig_complejidad = viz_system.crear_analisis_complejidad(datos_paciente)
            st.plotly_chart(fig_complejidad, use_container_width=True)
            
            # ✅ NUEVO: Añadir análisis clínico
            st.subheader("🔬 Análisis Clínico de la Predicción")
            
            # Realizar análisis clínico
            analisis_clinico = clinical_analyzer.analyze_prediction(datos_paciente, resultado)
            
            col_anal1, col_anal2 = st.columns(2)
            
            with col_anal1:
                st.write("**📋 Interpretación Clínica:**")
                st.info(f"**Duración:** {analisis_clinico['clinical_interpretation']['duration_category']}")
                st.info(f"**Confianza Clínica:** {analisis_clinico['clinical_confidence']:.0f}%")
                
                if analisis_clinico['risk_factors']:
                    st.warning("**⚠️ Factores de Riesgo Identificados:**")
                    for factor in analisis_clinico['risk_factors']:
                        st.write(f"• {factor}")
            
            with col_anal2:
                st.write("**💡 Recomendaciones:**")
                for recomendacion in analisis_clinico['treatment_recommendations']:
                    st.success(f"• {recomendacion}")
                
                if analisis_clinico['expected_challenges']:
                    st.write("**🎯 Desafíos Esperados:**")
                    for desafio in analisis_clinico['expected_challenges']:
                        st.info(f"• {desafio}")
            # Guardar paciente
            if nombre:
                paciente_data = {
                    'nombre': nombre,
                    'edad': edad,
                    'sexo': sexo,
                    'apiñamiento_mm': apiñamiento,
                    'sobremordida_mm': sobremordida,
                    'sobresalte_mm': sobresalte,
                    'observaciones': observaciones,
                    'duracion_predicha': resultado['prediccion'],
                    'duracion_min': resultado['intervalo_min'],
                    'duracion_max': resultado['intervalo_max'],
                    'modelo_usado': resultado['modelo_usado'],
                    'confianza_ml': resultado['confianza'],
                    'probabilidad_exito': probabilidad_exito, # ✅ Guardar nueva métrica
                    'estado': 'Nuevo',
                    'complejidad_score': utils.calculate_complexity_score(datos_paciente)  # ✅ NUEVO
                }
                
                try:
                    paciente_id = guardar_paciente(paciente_data)
                    st.success(f"📁 Paciente guardado con ID: ORTHO-{paciente_id}")
                    
                    # SECCIÓN DE REPORTE PDF MEJORADA
                    st.markdown("---")
                    st.subheader("📄 Generar Reporte PDF Profesional")
                    
                    if st.button("🖨️ Generar Reporte PDF Completo", key="pdf_new_patient"):
                        with st.spinner("Generando reporte PDF profesional..."):
                            # Preparar datos para el reporte
                            datos_reporte = {
                                **paciente_data,
                                'id': paciente_id,
                                'fecha_creacion': datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
                            }
                            
                            # Generar PDF
                            pdf_path = generar_reporte_pdf(datos_reporte, resultado)
                            
                            if pdf_path:
                                # Leer PDF generado
                                with open(pdf_path, "rb") as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                
                                # Botón de descarga
                                st.download_button(
                                    label="📥 Descargar Reporte PDF",
                                    data=pdf_bytes,
                                    file_name=f"reporte_orthopredict_{paciente_id}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf",
                                    help="Descargar reporte completo en formato PDF profesional",
                                    key="download_new_patient"
                                )
                                
                                # Limpiar archivo temporal
                                os.unlink(pdf_path)
                                
                                st.success("✅ Reporte PDF generado exitosamente!")
                                st.info("""
                                **📋 El reporte incluye:**
                                - Información completa del paciente
                                - Resultados de predicción detallados
                                - Parámetros clínicos con interpretación
                                - Análisis y recomendaciones personalizadas
                                - Gráficos visuales de predicción
                                - Plan de tratamiento específico
                                """)
                            else:
                                st.error("❌ Error generando el reporte PDF")
                
                except ValueError as e:
                    st.error(f"❌ Error validando datos: {e}")
                except Exception as e:
                    logger.error(f"Error guardando paciente: {e}")
                    st.error("❌ Error guardando paciente en la base de datos")

# ==================== PÁGINA: ENTRENAR MODELOS ====================
elif pagina == "📊 Entrenar Modelos":
    st.header("📊 Entrenamiento de Modelos ML con Datos Clínicos Realistas")
    
    col_train1, col_train2 = st.columns([2, 1])
    
    with col_train1:
        st.subheader("🧠 Entrenar con Datos Clínicos Realistas")
        
        with st.form("entrenamiento_clinico_form"):
            n_muestras = st.slider("Número de pacientes simulados", 500, 5000, 2000, 100)
            fuente_datos = st.radio("Fuente de datos:", 
                                  ["Datos Clínicos Realistas", "Datos Sintéticos Básicos"])
            
            tipo_modelo = st.selectbox("Tipo de modelo", 
                                     ["Ensemble Avanzado", "Modelo Basado en Reglas Clínicas"])
            
            if st.form_submit_button("🚀 Entrenar Modelo con Datos Clínicos"):
                with st.spinner("Generando datos clínicos y entrenando modelo..."):
                    
                    if fuente_datos == "Datos Clínicos Realistas":
                        # ✅ NUEVO: Usar datos clínicos realistas
                        modelo_entrenado = predictor_ml.entrenar_con_datos_clinicos_realistas(n_muestras)
                        
                        # Mostrar reporte de calidad de datos
                        quality_report = data_integration.get_data_quality_report()
                        with st.expander("📊 Reporte de Calidad de Datos"):
                            st.json(quality_report)
                            
                    else:
                        # Datos sintéticos básicos (comportamiento anterior)
                        datos_entrenamiento = predictor_ml.generar_datos_entrenamiento_avanzado(n_muestras)
                        if tipo_modelo == "Ensemble Avanzado":
                            modelo_entrenado = predictor_ml.entrenar_modelo_ensemble_hibrido(datos_entrenamiento)
                        else:
                            modelo_entrenado = predictor_ml.entrenar_modelo_backup(datos_entrenamiento)
                    
                    # ✅ NUEVO: Entrenar también el modelo de clasificación
                    if fuente_datos == "Datos Clínicos Realistas":
                        predictor_ml.entrenar_modelo_clasificacion_exito(data_integration.simulated_data)

                    # Guardar modelo
                    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
                    os.makedirs(models_dir, exist_ok=True)
                    model_base_path = os.path.join(models_dir, 'orthopredict')
                    predictor_ml.guardar_modelo_completo(model_base_path)
                    
                    st.success("✅ Modelo entrenado y guardado exitosamente!")
                    
                    # Mostrar métricas
                    if modelo_entrenado and 'metricas' in modelo_entrenado:
                        metricas = modelo_entrenado['metricas']
                        st.subheader("📈 Métricas del Modelo")
                        col_met1, col_met2, col_met3 = st.columns(3)
                        with col_met1:
                            st.metric("MAE", f"{metricas.get('mae', 'N/A')} meses")
                        with col_met2:
                            st.metric("MSE", f"{metricas.get('mse', 'N/A')}")
                        with col_met3:
                            st.metric("R²", f"{metricas.get('r2', 'N/A')}")
    
    with col_train2:
        st.subheader("🔬 Análisis de Datos Clínicos")
        
        if st.button("📊 Generar Dataset Clínico de Ejemplo"):
            with st.spinner("Generando dataset clínico realista..."):
                # ✅ NUEVO: Generar dataset clínico de ejemplo
                datos_ejemplo = clinical_simulator.generate_realistic_clinical_dataset(100)
                st.dataframe(datos_ejemplo.head(10), use_container_width=True)
                
                # Estadísticas clínicas del dataset
                st.write("**Estadísticas Clínicas:**")
                st.write(f"- Total pacientes: {len(datos_ejemplo)}")
                st.write(f"- Edad promedio: {datos_ejemplo['age'].mean():.1f} años")
                st.write(f"- Apiñamiento promedio: {datos_ejemplo['initial_crowding_mm'].mean():.1f} mm")
                st.write(f"- Duración promedio: {datos_ejemplo['actual_treatment_duration_months'].mean():.1f} meses")
                st.write(f"- Casos con extracciones: {datos_ejemplo['requires_extractions'].mean():.1%}")
        
        st.subheader("📋 Distribuciones Clínicas")
        if st.button("📈 Ver Análisis de Distribuciones"):
            # ✅ NUEVO: Análisis de distribuciones clínicas
            if data_integration.simulated_data is not None:
                df = data_integration.simulated_data
                
                # Gráfico de distribución de duraciones
                fig_duration = px.histogram(df, x='actual_treatment_duration_months', 
                                          title='Distribución de Duraciones de Tratamiento',
                                          nbins=15)
                st.plotly_chart(fig_duration, use_container_width=True)
                
                # Gráfico de complejidad vs duración
                fig_scatter = px.scatter(df, x='complexity_score', 
                                       y='actual_treatment_duration_months',
                                       color='malocclusion_type',
                                       title='Complejidad vs Duración del Tratamiento')
                st.plotly_chart(fig_scatter, use_container_width=True)

# ==================== PÁGINA: DIAGNÓSTICO DE MODELOS ====================
elif pagina == "🔍 Diagnóstico Modelos":
    st.header("🔍 Diagnóstico de Modelos y Análisis Clínico")
    
    # Pestañas para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Modelos ML", "📊 Datos Clínicos", "🎯 Análisis", "📈 Reportes"])
    
    with tab1:
        # Contenido existente de diagnóstico de modelos
        if not predictor_ml.ultimo_entrenamiento:
            st.warning("No hay modelos entrenados para diagnosticar")
        else:
            st.subheader("📊 Estado del Sistema ML")
            # ... (código existente) ...
    
    with tab2:
        st.subheader("📊 Análisis de Datos Clínicos")
        
        if st.button("🔄 Cargar Dataset Clínico", key="load_clinical_data"):
            with st.spinner("Generando dataset clínico realista..."):
                if data_integration.initialize_clinical_dataset(1000):
                    st.success("✅ Dataset clínico cargado exitosamente")
                else:
                    st.error("❌ Error cargando dataset clínico")
        
        if data_integration.simulated_data is not None:
            df = data_integration.simulated_data
            
            # Métricas clave del dataset
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pacientes", len(df))
            with col2:
                st.metric("Duración Promedio", f"{df['actual_treatment_duration_months'].mean():.1f} meses")
            with col3:
                st.metric("Apiñamiento Promedio", f"{df['initial_crowding_mm'].mean():.1f} mm")
            with col4:
                st.metric("Extracciones", f"{df['requires_extractions'].mean():.1%}")
            
            # Gráficos de análisis clínico
            st.subheader("📈 Distribuciones Clínicas")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Distribución de edades
                fig_age = px.histogram(df, x='age', nbins=15, 
                                     title='Distribución de Edades')
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col_chart2:
                # Distribución de complejidad
                fig_comp = px.histogram(df, x='complexity_score', nbins=15,
                                      title='Distribución de Scores de Complejidad')
                st.plotly_chart(fig_comp, use_container_width=True)
            
            # Análisis por tipo de maloclusión
            st.subheader("🎯 Análisis por Tipo de Maloclusión")
            malocclusion_stats = df.groupby('malocclusion_type').agg({
                'actual_treatment_duration_months': ['mean', 'std', 'count'],
                'complexity_score': 'mean'
            }).round(2)
            
            st.dataframe(malocclusion_stats, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Análisis de Performance Clínico")
        
        if data_integration.simulated_data is not None and 'principal' in predictor_ml.modelos:
            # Evaluar modelo en datos clínicos
            if st.button("📊 Evaluar Modelo en Datos Clínicos"):
                with st.spinner("Evaluando modelo en dataset clínico..."):
                    training_data = data_integration.get_training_data()
                    
                    if not training_data.empty:
                        # Realizar evaluación
                        X_test = training_data.drop('actual_treatment_duration_months', axis=1)
                        y_test = training_data['actual_treatment_duration_months']
                        
                        # Calcular métricas
                        predictions = []
                        for _, row in X_test.iterrows():
                            # Convertir a formato de paciente
                            paciente = {
                                'edad': row['age'],
                                'sexo': 'F' if row['gender_encoded'] == 0 else 'M',
                                'apiñamiento_mm': row['initial_crowding_mm'],
                                'sobremordida_mm': row.get('initial_overbite_mm', 2.5),
                                'sobresalte_mm': row.get('initial_overjet_mm', 2.5)
                            }
                            pred = predictor_ml.predecir_duracion(paciente)
                            predictions.append(pred['prediccion'])
                        
                        # Calcular métricas
                        mae = np.mean(np.abs(y_test - predictions))
                        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
                        r2 = 1 - (np.var(y_test - predictions) / np.var(y_test))
                        
                        col_met1, col_met2, col_met3 = st.columns(3)
                        with col_met1:
                            st.metric("MAE Clínico", f"{mae:.2f} meses")
                        with col_met2:
                            st.metric("RMSE Clínico", f"{rmse:.2f} meses")
                        with col_met3:
                            st.metric("R² Clínico", f"{r2:.3f}")
    
    with tab4:
        st.subheader("📈 Reportes Clínicos")
        
        if st.button("📋 Generar Reporte Clínico Completo"):
            if data_integration.simulated_data is not None:
                # Generar reporte clínico
                report_data = {
                    'fecha_generacion': datetime.datetime.now().isoformat(),
                    'total_pacientes': len(data_integration.simulated_data),
                    'metricas_generales': data_integration.simulated_data.describe().to_dict(),
                    'distribucion_maloclusiones': data_integration.simulated_data['malocclusion_type'].value_counts().to_dict(),
                    'calidad_datos': data_integration.get_data_quality_report()
                }
                
                with st.expander("📊 Reporte Clínico Detallado"):
                    st.json(report_data)
                
                # Exportar a Excel
                if st.button("📊 Exportar a Excel"):
                    excel_path = "reporte_clinico_orthopredict.xlsx"
                    if utils.export_to_excel(data_integration.simulated_data, excel_path):
                        st.success("✅ Reporte exportado a Excel")
                    else:
                        st.error("❌ Error exportando reporte")

# ==================== PÁGINA: GESTIÓN PACIENTES OPTIMIZADA ====================
elif pagina == "👥 Gestión Pacientes":
    st.header("👥 Gestión de Pacientes Optimizada")
    
    pacientes = obtener_todos_pacientes()
    
    if not pacientes:
        st.info("""
        📝 **No hay pacientes registrados aún**
        
        Para comenzar:
        1. Ve a **🎯 Predictor ML**
        2. Ingresa los datos de un paciente
        3. La información se guardará automáticamente
        """)
    else:
        col_top1, col_top2, col_top3, col_top4 = st.columns([2, 1, 1, 1])  # ✅ NUEVA columna
        
        with col_top1:
            st.subheader(f"📋 Pacientes Registrados: {len(pacientes)}")
        
        with col_top2:
            # ✅ NUEVO: Exportar a Excel optimizado
            if st.button("📊 Exportar a Excel", key="export_excel"):
                with st.spinner("Generando archivo Excel..."):
                    try:
                        # Crear archivo temporal
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                            excel_path = tmp_file.name
                        
                        # Usar utilidad optimizada
                        # Pasamos la lista de pacientes directamente
                        if backup_system.export_to_excel(pacientes, excel_path):
                            with open(excel_path, "rb") as f:
                                excel_data = f.read()
                            
                            st.download_button(
                                label="📥 Descargar Excel",
                                data=excel_data,
                                file_name=f"pacientes_orthopredict_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_excel"
                            )
                            os.unlink(excel_path)
                        else:
                            st.error("❌ Error generando archivo Excel")
                    except Exception as e:
                        logger.error(f"Error exportando a Excel: {e}")
                        st.error("❌ Error en exportación Excel")
        
        with col_top3:
            # Exportar a CSV
            if st.button("📄 Exportar a CSV"):
                df_pacientes = pd.DataFrame(pacientes)
                csv = df_pacientes.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv,
                    file_name=f"pacientes_orthopredict_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col_top4:
            # ✅ NUEVO: Estadísticas de cache
            if st.button("⚡ Cache Stats", key="cache_stats"):
                stats = cache_system.get_stats()
                with st.expander("Estadísticas de Cache"):
                    st.json(stats)
        
        st.markdown("---")

        # ✅ CORRECCIÓN: Añadir búsqueda y listado de pacientes
        search_query = st.text_input("🔍 Buscar paciente por nombre o ID", "")

        if search_query:
            search_query = search_query.lower()
            pacientes_filtrados = [
                p for p in pacientes 
                if search_query in p.get('nombre', '').lower() or str(p.get('id', '')) == search_query
            ]
        else:
            pacientes_filtrados = pacientes

        # Mostrar pacientes en un formato de lista
        for paciente in sorted(pacientes_filtrados, key=lambda p: p.get('id', 0), reverse=True):
            with st.expander(f"**ID: ORTHO-{paciente.get('id', 'N/A')}** - {paciente.get('nombre', 'Sin Nombre')}"):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write("**Datos Clínicos:**")
                    st.write(f"- **Edad:** {paciente.get('edad', 'N/A')} años")
                    st.write(f"- **Sexo:** {paciente.get('sexo', 'N/A')}")
                    st.write(f"- **Apiñamiento:** {paciente.get('apiñamiento_mm', 'N/A')} mm")
                    st.write(f"- **Sobremordida:** {paciente.get('sobremordida_mm', 'N/A')} mm")
                    st.write(f"- **Sobresalte:** {paciente.get('sobresalte_mm', 'N/A')} mm")

                with col2:
                    st.write("**Resultados de Predicción:**")
                    st.write(f"- **Duración Predicha:** {paciente.get('duracion_predicha', 'N/A')} meses")
                    st.write(f"- **Confianza ML:** {paciente.get('confianza_ml', 'N/A')}")
                    
                    # ✅ CORRECCIÓN: Formatear solo si es un número
                    prob_exito = paciente.get('probabilidad_exito')
                    prob_exito_str = f"{prob_exito:.0%}" if isinstance(prob_exito, (int, float)) else "N/A"
                    st.write(f"- **Prob. Éxito:** {prob_exito_str}")
                    st.write(f"- **Modelo Usado:** {paciente.get('modelo_usado', 'N/A').replace('_', ' ').title()}")

                with col3:
                    st.write("**Acciones:**")
                    if st.button("📄 Generar Reporte PDF", key=f"pdf_{paciente.get('id')}"):
                        with st.spinner("Generando reporte..."):
                            prediccion_data = {
                                'prediccion': paciente.get('duracion_predicha'),
                                'intervalo_min': paciente.get('duracion_min'),
                                'intervalo_max': paciente.get('duracion_max'),
                                'confianza': paciente.get('confianza_ml'),
                                'modelo_usado': paciente.get('modelo_usado')
                            }
                            pdf_path = generar_reporte_pdf(paciente, prediccion_data)
                            if pdf_path:
                                with open(pdf_path, "rb") as f:
                                    st.download_button(
                                        "📥 Descargar Reporte",
                                        f,
                                        file_name=f"reporte_orthopredict_{paciente.get('id')}.pdf",
                                        mime="application/pdf"
                                    )
                                os.unlink(pdf_path)
                            else:
                                st.error("❌ Error al generar el PDF.")

        # ✅ NUEVO: Análisis comparativo con gráficos interactivos
        with st.expander("📊 Análisis Comparativo Avanzado", expanded=False):
            if len(pacientes) >= 2:
                df = pd.DataFrame(pacientes)

                # ✅ CORRECCIÓN: Asegurar que la columna 'complejidad_score' exista
                if 'complejidad_score' not in df.columns:
                    df['complejidad_score'] = df.apply(utils.calculate_complexity_score, axis=1)
                    st.info("ℹ️ Se calculó el score de complejidad para pacientes antiguos.")
                
                # Gráfico interactivo de distribución
                st.subheader("Distribución de Pacientes por Complejidad")
                fig_dist = px.histogram(df, x='complejidad_score', 
                                      title='Distribución de Scores de Complejidad',
                                      color_discrete_sequence=['#2E86AB'])
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Scatter plot interactivo
                st.subheader("Relación Apiñamiento vs Duración Predicha")
                
                # ✅ CORRECCIÓN: Eliminar filas con NaN en columnas clave para evitar errores en 'size'
                df_clean = df.dropna(subset=['apiñamiento_mm', 'duracion_predicha', 'complejidad_score'])
                
                fig_scatter = px.scatter(df_clean, x='apiñamiento_mm', y='duracion_predicha',
                                       color='complejidad_score',
                                       size='complejidad_score',
                                       hover_data=['edad', 'sexo'],
                                       title='Apiñamiento vs Duración Predicha')
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            else:
                st.info("Se necesitan al menos 2 pacientes para el análisis comparativo")

# ==================== PÁGINA: ANÁLISIS DE COHORTES ====================
elif pagina == "🔬 Análisis de Cohortes":
    st.header("🔬 Análisis Avanzado de Cohortes de Pacientes")
    st.info("Utiliza los filtros en la barra lateral para segmentar y analizar tu base de datos de pacientes.")

    pacientes = obtener_todos_pacientes()

    if not pacientes:
        st.warning("No hay pacientes registrados para analizar.")
    else:
        df = pd.DataFrame(pacientes)

        # Asegurar que la columna 'complejidad_score' exista
        if 'complejidad_score' not in df.columns:
            df['complejidad_score'] = df.apply(utils.calculate_complexity_score, axis=1)

        # --- Filtros en la Sidebar ---
        st.sidebar.header("Filtros de Cohorte")
        
        # Filtro por edad
        edad_min = int(df['edad'].min())
        edad_max = int(df['edad'].max())
        if edad_min == edad_max:
            edad_max += 1 # Asegurar que el rango sea válido
        edad_seleccionada = st.sidebar.slider("Rango de Edad", edad_min, edad_max, (edad_min, edad_max))

        # Filtro por sexo
        sexo_opciones = df['sexo'].unique().tolist()
        sexo_seleccionado = st.sidebar.multiselect("Sexo", sexo_opciones, default=sexo_opciones)

        # Filtro por complejidad
        # ✅ CORRECCIÓN: Manejar el caso donde min y max son iguales
        complejidad_min = int(df['complejidad_score'].min())
        complejidad_max = int(df['complejidad_score'].max())
        if complejidad_min == complejidad_max:
            complejidad_max += 1 # Asegurar que el rango sea válido
        complejidad_seleccionada = st.sidebar.slider("Rango de Complejidad (Score)", complejidad_min, complejidad_max, (complejidad_min, complejidad_max))

        # Aplicar filtros
        df_filtrado = df[
            (df['edad'].between(edad_seleccionada[0], edad_seleccionada[1])) &
            (df['sexo'].isin(sexo_seleccionado)) &
            (df['complejidad_score'].between(complejidad_seleccionada[0], complejidad_seleccionada[1]))
        ]

        st.subheader(f"Análisis de la cohorte seleccionada ({len(df_filtrado)} pacientes)")

        if df_filtrado.empty:
            st.warning("Ningún paciente coincide con los filtros seleccionados.")
        else:
            # Métricas de la cohorte
            col1, col2, col3 = st.columns(3)
            col1.metric("Duración Promedio", f"{df_filtrado['duracion_predicha'].mean():.1f} meses")
            col2.metric("Complejidad Promedio", f"{df_filtrado['complejidad_score'].mean():.1f}")
            col3.metric("Edad Promedio", f"{df_filtrado['edad'].mean():.1f} años")

            st.markdown("---")

            # Gráficos avanzados
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.write("**Distribución de Duración por Complejidad**")
                df_filtrado['grupo_complejidad'] = pd.cut(df_filtrado['complejidad_score'], bins=[0, 40, 70, 100], labels=['Baja', 'Media', 'Alta'])
                fig_box = px.box(df_filtrado, x='grupo_complejidad', y='duracion_predicha', color='grupo_complejidad', title="Duración vs. Nivel de Complejidad")
                st.plotly_chart(fig_box, use_container_width=True)
            with col_g2:
                st.write("**Composición de la Cohorte**")
                df_filtrado['grupo_edad'] = pd.cut(df_filtrado['edad'], bins=[18, 25, 35, 50], labels=['18-25', '26-35', '36+'])
                fig_sunburst = px.sunburst(df_filtrado.dropna(subset=['grupo_edad']), path=['sexo', 'grupo_edad'], title="Distribución por Sexo y Edad")
                st.plotly_chart(fig_sunburst, use_container_width=True)

# ==================== PÁGINA: MONITOREO ML ====================
elif pagina == "📈 Monitoreo ML":
    st.header("📈 Monitoreo de Modelos ML en Tiempo Real")
    
    # ✅ CORRECCIÓN: Forzar una evaluación inicial si el sistema está inactivo
    if not sistema_monitoreo.metricas_historicas:
        with st.spinner("Inicializando sistema de monitoreo..."):
            pacientes = obtener_todos_pacientes()
            if len(pacientes) >= 10:
                sistema_monitoreo.evaluar_modelo_actual(predictor_ml, pd.DataFrame(pacientes))

    # ✅ NUEVO: Usar el dashboard avanzado
    monitoring_dashboard.render_dashboard()
    
    # Mantener la funcionalidad existente como fallback
    with st.expander("🔧 Configuración Avanzada del Monitoreo", expanded=False):
        # Inicializar sistema si no está activo
        if st.button("🔄 Inicializar Sistema de Monitoreo"):
            if inicializar_sistema_monitoreo():
                st.success("✅ Sistema de monitoreo inicializado")
                st.rerun()
            else:
                st.error("❌ Error inicializando monitoreo")
        
        # Obtener datos de monitoreo tradicional
        datos_monitoreo = obtener_datos_monitoreo()
        
        if datos_monitoreo.get('estado_sistema') == 'ACTIVO':
            st.info("Sistema de monitoreo tradicional activo")
        else:
            st.warning("Sistema de monitoreo tradicional no inicializado")

# ==================== PÁGINA: SISTEMA OPTIMIZADO ====================
elif pagina == "⚙️ Sistema":
    st.header("⚙️ Sistema y Configuración Optimizado")
    
    st.subheader("📁 Estado del Sistema Avanzado")
    
    col_sys1, col_sys2 = st.columns(2)
    
    with col_sys1:
        st.write("**Archivos del sistema:**")
        
        archivos = {
            "Base de datos pacientes": os.path.exists(config.get_database_path()),
            "Directorio de modelos": os.path.exists(config.MODELS_DIR),
            "Directorio data": os.path.exists(config.DATA_DIR),
            "Directorio logs": os.path.exists(config.LOGS_DIR),
            "Directorio backups": os.path.exists(config.BACKUP_DIR),
            "Módulo PDF": True,
            "Módulo Monitoreo": True,
            "Sistema de Cache": True,
            "Sistema de Visualización": True
        }
        
        if AUTH_ENABLED:
            archivos["Sistema de autenticación"] = os.path.exists("users.json")
            archivos["Sistema de backup"] = os.path.exists("backups")
        
        for archivo, existe in archivos.items():
            if existe:
                st.success(f"✅ {archivo}")
            else:
                st.error(f"❌ {archivo}")
    
    with col_sys2:
        st.write("**Información ML Optimizada:**")
        if predictor_ml.ultimo_entrenamiento:
            st.success(f"Modelo entrenado: {predictor_ml.ultimo_entrenamiento.strftime('%d/%m/%Y %H:%M')}")
        else:
            st.warning("Modelo no entrenado")
        
        st.write(f"Modelos disponibles: {len(predictor_ml.modelos)}")
        st.write(f"Pacientes en base: {len(obtener_todos_pacientes())}")
        
        # ✅ NUEVO: Estadísticas de cache
        cache_stats = cache_system.get_stats()
        st.write(f"Tasa de aciertos cache: {cache_stats['hit_rate']}%")
        st.write(f"Total operaciones cache: {cache_stats['total_hits'] + cache_stats['total_misses']}")
        
        if AUTH_ENABLED:
            st.write(f"Usuarios registrados: {len(auth_system.get_all_users())}")
    
    # ✅ NUEVO: Configuración del sistema
    with st.expander("🔧 Configuración del Sistema", expanded=False):
        # ✅ CORRECCIÓN: Mostrar la configuración de forma más legible
        config_dict = config.to_dict()
        st.write("Valores de configuración actualmente cargados en la aplicación:")
        
        for key, value in config_dict.items():
            # Formatear la clave para que sea más legible
            formatted_key = key.replace('_', ' ').title()
            st.write(f"**{formatted_key}:** `{value}`")
    
    # ✅ NUEVO: Gestión de cache
    with st.expander("⚡ Gestión de Cache", expanded=False):
        col_cache1, col_cache2 = st.columns(2)
        
        with col_cache1:
            if st.button("🔄 Actualizar Cache Stats"):
                st.rerun()
            
            if st.button("🗑️ Limpiar Todo el Cache"):
                cache_system.clear()
                st.success("✅ Cache limpiado exitosamente")
                st.rerun()
        
        with col_cache2:
            st.write("**Estadísticas Actuales:**")
            stats = cache_system.get_stats()

# ==================== PÁGINA: ADMINISTRACIÓN (SOLO ADMIN) ====================
elif pagina == "👑 Administración":
    if AUTH_ENABLED:
        require_role("admin")  # Doble seguridad
        st.header("👑 Panel de Administración del Sistema")
        
        # Pestañas para organizar las herramientas de administración
        admin_tabs = st.tabs(["💾 Gestión de Backups", "👥 Gestión de Usuarios", "🔧 Configuración", " Visor de Logs"])
        
        with admin_tabs[0]:
            st.subheader("💾 Gestión de Backups de la Base de Datos")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.info("Crea o restaura copias de seguridad de la base de datos de pacientes.")
                
                if st.button("➕ Crear Backup Manual Ahora", type="primary"):
                    with st.spinner("Creando backup..."):
                        db_data = inicializar_base_datos()
                        backup_file = backup_system.create_backup(db_data, "manual")
                        if backup_file:
                            st.success(f"✅ Backup creado: {os.path.basename(backup_file)}")
                        else:
                            st.error("❌ Error al crear el backup.")
                
                # Estadísticas de backups
                st.markdown("---")
                st.subheader("📊 Estadísticas")
                stats = backup_system.get_backup_stats()
                st.metric("Backups Manuales", stats['manual_backups'])
                st.metric("Backups Automáticos", stats['auto_backups'])
                st.metric("Tamaño Total", f"{stats['total_size_mb']:.2f} MB")

            with col2:
                st.subheader("📋 Backups Disponibles")
                backups = backup_system.list_backups()
                
                if not backups:
                    st.info("No hay backups disponibles.")
                else:
                    for backup in backups:
                        with st.container(border=True):
                            col_b1, col_b2, col_b3 = st.columns([3, 1, 1])
                            with col_b1:
                                st.write(f"**Archivo:** `{os.path.basename(backup['file'])}`")
                                st.write(f"**Fecha:** {backup['date']} | **Tipo:** {backup['type']}")
                            with col_b2:
                                if st.button("🔄 Restaurar", key=f"restore_{backup['file']}", help="Restaurar la base de datos a este punto. ¡Esta acción no se puede deshacer!"):
                                    # Advertencia de confirmación
                                    st.session_state.backup_to_restore = backup['file']
                            with col_b3:
                                with open(backup['file'], "rb") as f:
                                    st.download_button("📥 Descargar", f, file_name=os.path.basename(backup['file']), key=f"download_{backup['file']}")
            
            # Lógica de confirmación para restaurar
            if 'backup_to_restore' in st.session_state and st.session_state.backup_to_restore:
                st.warning(f"**⚠️ ¿Estás seguro de que quieres restaurar el backup `{os.path.basename(st.session_state.backup_to_restore)}`?** Esta acción sobreescribirá la base de datos actual.")
                if st.button("Sí, restaurar", type="primary"):
                    if backup_system.restore_backup(st.session_state.backup_to_restore):
                        st.success("✅ Base de datos restaurada exitosamente. La aplicación se recargará.")
                        cache_system.clear('patient_data') # Limpiar cache de pacientes
                        st.session_state.backup_to_restore = None
                        st.rerun()
                    else:
                        st.error("❌ Error durante la restauración.")
                if st.button("Cancelar"):
                    st.session_state.backup_to_restore = None
                    st.rerun()

        with admin_tabs[1]:
            st.subheader("👥 Gestión de Usuarios")
            all_users = auth_system.get_all_users()

            # Formulario para crear un nuevo usuario
            with st.expander("➕ Crear Nuevo Usuario"):
                with st.form("create_user_form", clear_on_submit=True):
                    st.write("Rellena los datos para crear un nuevo usuario.")
                    new_username = st.text_input("Nombre de usuario*", help="El identificador para iniciar sesión.")
                    new_password = st.text_input("Contraseña*", type="password")
                    new_name = st.text_input("Nombre Completo")
                    new_email = st.text_input("Email")
                    new_role = st.selectbox("Rol", ["doctor", "admin"])

                    if st.form_submit_button("Crear Usuario"):
                        if new_username and new_password:
                            if auth_system.create_user(new_username, new_password, new_role, new_name, new_email):
                                st.success(f"✅ Usuario '{new_username}' creado exitosamente.")
                                st.rerun()
                            else:
                                st.error(f"❌ El usuario '{new_username}' ya existe.")
                        else:
                            st.warning("El nombre de usuario y la contraseña son obligatorios.")

            st.markdown("---")
            st.subheader("📋 Usuarios Registrados")

            # Listar y gestionar usuarios existentes
            for username, user_data in all_users.items():
                with st.container(border=True):
                    role_badge = "👑" if user_data['role'] == 'admin' else "👨‍⚕️"
                    st.write(f"**{role_badge} {user_data['name']}** (`{username}`)")
                    
                    with st.expander("Ver y Editar Detalles"):
                        # Formulario para editar información
                        with st.form(f"edit_form_{username}"):
                            new_name_edit = st.text_input("Nombre Completo", value=user_data['name'], key=f"name_{username}")
                            new_email_edit = st.text_input("Email", value=user_data['email'], key=f"email_{username}")
                            new_role_edit = st.selectbox("Rol", ["doctor", "admin"], index=0 if user_data['role'] == 'doctor' else 1, key=f"role_{username}")
                            
                            if st.form_submit_button("Guardar Cambios"):
                                auth_system.update_user(username, name=new_name_edit, email=new_email_edit, role=new_role_edit)
                                st.success("✅ Información actualizada.")
                                st.rerun()

                        # Botón para eliminar (protegido)
                        if username != "admin": # Proteger al admin principal
                            if st.button("🗑️ Eliminar Usuario", key=f"delete_{username}", type="secondary"):
                                if auth_system.delete_user(username):
                                    st.success(f"✅ Usuario '{username}' eliminado.")
                                    st.rerun()
                                else:
                                    st.error("❌ No se pudo eliminar el usuario.")

        with admin_tabs[2]:
            st.subheader("🔧 Configuración del Sistema")
            st.info("Modifica la configuración del sistema directamente. Se requiere un reinicio para aplicar los cambios.")

            # Encontrar el archivo .env
            dotenv_file = find_dotenv()
            if not os.path.exists(dotenv_file):
                # Crear si no existe
                with open(".env", "w") as f:
                    f.write("# Archivo de configuración de OrthoPredict\n")
                dotenv_file = find_dotenv()

            with st.form("system_config_form"):
                st.write("**Configuración General y Debug**")
                cfg_debug = st.toggle("Modo Debug", value=config.DEBUG, help="Activa logs detallados y otras ayudas para desarrolladores.")
                cfg_app_version = st.text_input("Versión de la App", value=config.APP_VERSION)

                st.markdown("---")
                st.write("**Configuración de Cache**")
                cfg_cache_enabled = st.toggle("Habilitar Cache", value=config.CACHE_ENABLED, help="Mejora el rendimiento guardando resultados frecuentes.")
                cfg_cache_ttl = st.number_input("TTL del Cache (segundos)", min_value=60, value=config.CACHE_TTL, help="Tiempo que los elementos permanecen en cache.")

                st.markdown("---")
                st.write("**Configuración de Seguridad**")
                cfg_session_timeout = st.number_input("Timeout de Sesión (segundos)", min_value=300, value=config.SESSION_TIMEOUT, help="Tiempo de inactividad antes de cerrar la sesión.")
                cfg_max_logins = st.number_input("Máximos Intentos de Login", min_value=3, max_value=10, value=config.MAX_LOGIN_ATTEMPTS)

                if st.form_submit_button("💾 Guardar Configuración", type="primary"):
                    set_key(dotenv_file, "ORTHOPREDICT_DEBUG", str(cfg_debug).lower())
                    set_key(dotenv_file, "ORTHOPREDICT_APP_VERSION", cfg_app_version)
                    set_key(dotenv_file, "ORTHOPREDICT_CACHE_ENABLED", str(cfg_cache_enabled).lower())
                    set_key(dotenv_file, "ORTHOPREDICT_CACHE_TTL", str(cfg_cache_ttl))
                    set_key(dotenv_file, "ORTHOPREDICT_SESSION_TIMEOUT", str(cfg_session_timeout))
                    set_key(dotenv_file, "ORTHOPREDICT_MAX_LOGIN_ATTEMPTS", str(cfg_max_logins))
                    st.success("✅ Configuración guardada en el archivo .env.")
                    st.warning("⚠️ Por favor, reinicia la aplicación para que los cambios surtan efecto.")
        
        with admin_tabs[3]:
            st.subheader("📜 Visor de Logs de la Aplicación")
            log_file_path = os.path.join(config.LOGS_DIR, 'orthopredict.log')

            if not os.path.exists(log_file_path):
                st.warning("El archivo de log no existe todavía. Se creará con la primera actividad.")
            else:
                col1, col2, col3 = st.columns([1,1,2])
                with col1:
                    log_level = st.selectbox("Filtrar por Nivel", ["TODOS", "INFO", "WARNING", "ERROR"])
                with col2:
                    num_lines = st.slider("Líneas a mostrar", 50, 1000, 200, 50)
                
                if st.button("🔄 Refrescar Logs"):
                    st.rerun()

                try:
                    # ✅ CORRECCIÓN: Añadir errors='ignore' para evitar fallos con caracteres mal codificados
                    with open(log_file_path, "r", encoding="utf-8", errors='ignore') as f:
                        lines = f.readlines()
                    
                    # Filtrar y mostrar las últimas N líneas
                    filtered_lines = lines
                    if log_level != "TODOS":
                        filtered_lines = [line for line in lines if f" - {log_level} - " in line]
                    
                    st.code("".join(filtered_lines[-num_lines:]), language="log")

                    # Botón de descarga
                    st.download_button("📥 Descargar Log Completo", "".join(lines), "orthopredict.log")

                except Exception as e:
                    st.error(f"❌ No se pudo leer el archivo de log: {e}")

    else:
        st.warning("El sistema de autenticación no está habilitado.")

# ==================== FOOTER ACTUALIZADO ====================
st.markdown("---")
footer_cols = st.columns(3)

with footer_cols[0]:
    st.markdown(
        f"🤖 **{config.APP_NAME} v{config.APP_VERSION}** | "
        "Sistema Optimizado con ML"
    )

with footer_cols[1]:
    st.markdown(
        "📊 Visualizaciones Interactivas | "
        "⚡ Cache Inteligente"
    )

with footer_cols[2]:
    if AUTH_ENABLED:
        st.markdown("🔐 Autenticación Segura | 💾 Backup Automático")
    else:
        st.markdown("⚡ Sistema Optimizado | 🎯 Predicciones Precisas")