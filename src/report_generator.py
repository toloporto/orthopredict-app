# orthopredict_app/src/report_generator.py
import os
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
import io
import base64

# ✅ NUEVO: Importar sistemas optimizados
from visualization_system import viz_system
from config import config
from utils import utils
import logging

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Crear estilos personalizados para el reporte"""
        # Estilo para título principal
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB'),
            alignment=1,  # Centrado
            fontName='Helvetica-Bold'
        ))
        
        # Estilo para subtítulos
        self.styles.add(ParagraphStyle(
            name='SubTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=colors.HexColor('#2E86AB'),
            fontName='Helvetica-Bold'
        ))
        
        # Estilo para información del paciente
        self.styles.add(ParagraphStyle(
            name='PatientInfo',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.HexColor('#333333'),
            fontName='Helvetica'
        ))
        
        # Estilo para resultados
        self.styles.add(ParagraphStyle(
            name='Results',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.HexColor('#2E86AB'),
            backColor=colors.HexColor('#F8F9FA'),
            fontName='Helvetica-Bold'
        ))
        
        # Estilo para destacados
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#D32F2F'),
            backColor=colors.HexColor('#FFEBEE'),
            fontName='Helvetica-Bold'
        ))

    def generar_reporte_paciente(self, paciente_data, prediccion_data, output_path):
        """Generar reporte PDF completo para un paciente"""
        try:
            # Crear documento
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=50,
                leftMargin=50,
                topMargin=50,
                bottomMargin=50
            )
            
            story = []
            
            # 1. ENCABEZADO
            story.extend(self._crear_encabezado())
            
            # 2. RESUMEN EJECUTIVO
            story.extend(self._crear_resumen_ejecutivo(paciente_data, prediccion_data))
            
            # 3. INFORMACIÓN DEL PACIENTE
            story.extend(self._crear_seccion_paciente(paciente_data))
            
            # 4. RESULTADOS DE PREDICCIÓN
            story.extend(self._crear_seccion_prediccion(prediccion_data))
            
            # 5. GRÁFICOS PLOTLY INCORPORADOS
            story.extend(self._crear_graficos_plotly(paciente_data, prediccion_data))
            
            # 6. PARÁMETROS CLÍNICOS DETALLADOS
            story.extend(self._crear_seccion_parametros_detallados(paciente_data))

            # 7. ANÁLISIS Y RECOMENDACIONES
            story.extend(self._crear_seccion_analisis_completo(paciente_data, prediccion_data))
            
            # 8. PIE DE PÁGINA
            story.extend(self._crear_pie_pagina())
            
            # Construir PDF
            try:
                doc.build(story)
                logger.info(f"✅ Reporte PDF generado exitosamente: {output_path}")
                return True
            except AttributeError as e:
                if "'list' object has no attribute 'getKeepWithNext'" in str(e):
                    return self._generar_reporte_alternativo(paciente_data, prediccion_data, output_path)
                else:
                    raise e
            
        except Exception as e:
            logger.error(f"Error generando PDF: {e}", exc_info=True)
            return False

    def _generar_reporte_alternativo(self, paciente_data, prediccion_data, output_path):
        """Método alternativo para generar PDF sin usar listas complejas"""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            
            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            
            # Encabezado
            c.setFont("Helvetica-Bold", 16)
            c.setFillColor(colors.HexColor('#2E86AB'))
            c.drawString(50, height - 50, "ORTHOPREDICT PRO ML")
            c.setFont("Helvetica", 10)
            c.setFillColor(colors.black)
            c.drawString(50, height - 70, "Sistema de Inteligencia Artificial para Ortodoncia")
            c.drawString(50, height - 85, f"Reporte Generado: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
            
            # Información del paciente
            y_position = height - 120
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_position, "INFORMACIÓN DEL PACIENTE")
            y_position -= 20
            
            c.setFont("Helvetica", 10)
            info_lines = [
                f"Nombre: {paciente_data.get('nombre', 'No especificado')}",
                f"Edad: {paciente_data.get('edad', 'N/A')} años",
                f"Sexo: {paciente_data.get('sexo', 'N/A')}",
                f"ID: ORTHO-{paciente_data.get('id', 'N/A')}"
            ]
            
            for line in info_lines:
                c.drawString(50, y_position, line)
                y_position -= 15
            
            # Resultados de predicción
            y_position -= 10
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_position, "RESULTADOS DE PREDICCIÓN")
            y_position -= 20
            
            c.setFont("Helvetica", 10)
            resultado_lines = [
                f"Duración Estimada: {prediccion_data.get('prediccion', 'N/A')} meses",
                f"Rango Probable: {prediccion_data.get('intervalo_min', 'N/A')} - {prediccion_data.get('intervalo_max', 'N/A')} meses",
                f"Confianza: {prediccion_data.get('confianza', 'N/A')}%",
                f"Modelo: {prediccion_data.get('modelo_usado', 'N/A').replace('_', ' ').title()}"
            ]
            
            for line in resultado_lines:
                c.drawString(50, y_position, line)
                y_position -= 15
            
            c.save()
            logger.info(f"✅ Reporte PDF alternativo generado: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error en método alternativo: {e}", exc_info=True)
            return False

    def _crear_encabezado(self):
        """Crear encabezado del reporte"""
        elements = []
        elements.append(Paragraph("ORTHOPREDICT PRO ML", self.styles['MainTitle']))
        elements.append(Paragraph("Sistema de Inteligencia Artificial para Ortodoncia", self.styles['Normal']))
        elements.append(Spacer(1, 10))
        elements.append(self._crear_linea_divisoria(colors.HexColor('#2E86AB'), 2))
        elements.append(Spacer(1, 15))
        
        info_table = [
            [Paragraph("<b>Reporte Generado:</b>", self.styles['PatientInfo']), 
             Paragraph(datetime.datetime.now().strftime('%d/%m/%Y %H:%M'), self.styles['PatientInfo'])],
            [Paragraph("<b>Tipo de Reporte:</b>", self.styles['PatientInfo']), 
             Paragraph("Análisis Predictivo de Tratamiento", self.styles['PatientInfo'])],
            [Paragraph("<b>Versión del Sistema:</b>", self.styles['PatientInfo']), 
             Paragraph(f"{config.APP_NAME} v{config.APP_VERSION}", self.styles['PatientInfo'])]
        ]
        
        tabla_info = Table(info_table, colWidths=[2*inch, 3*inch])
        tabla_info.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        elements.append(tabla_info)
        elements.append(Spacer(1, 20))
        return elements

    def _crear_resumen_ejecutivo(self, paciente_data, prediccion_data):
        """Crear resumen ejecutivo con los puntos clave"""
        elements = []
        elements.append(Paragraph("📊 RESUMEN EJECUTIVO", self.styles['SubTitle']))
        
        duracion_pred = prediccion_data.get('prediccion', 0)
        confianza = prediccion_data.get('confianza', 0)
        complejidad = self._calcular_nivel_complejidad(paciente_data.get('apiñamiento_mm', 0))
        
        resumen_cards = [
            [self._crear_tarjeta("Duración Estimada", f"{duracion_pred} meses", "#4CAF50"),
             self._crear_tarjeta("Nivel de Confianza", f"{confianza}%", "#2196F3")],
            [self._crear_tarjeta("Complejidad del Caso", complejidad, "#FF9800"),
             self._crear_tarjeta("Modelo Utilizado", prediccion_data.get('modelo_usado', 'N/A').replace('_', ' ').title(), "#9C27B0")]
        ]
        
        tabla_resumen = Table(resumen_cards, colWidths=[3*inch, 3*inch])
        tabla_resumen.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
        elements.append(tabla_resumen)
        elements.append(Spacer(1, 15))
        return elements

    def _crear_tarjeta(self, titulo, valor, color):
        """Crear una tarjeta visual para el resumen"""
        return Table([
            [Paragraph(f"<b>{titulo}</b>", self.styles['PatientInfo'])],
            [Paragraph(f"<font size='14' color='{color}'><b>{valor}</b></font>", self.styles['PatientInfo'])]
        ], colWidths=[2.8*inch], rowHeights=[0.4*inch, 0.6*inch]).setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(color)),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F5F5F5')),
        ]))

    def _crear_seccion_paciente(self, paciente_data):
        """Crear sección de información del paciente"""
        elements = []
        elements.append(Paragraph("👤 INFORMACIÓN DEL PACIENTE", self.styles['SubTitle']))
        
        paciente_info = [
            [Paragraph("<b>Nombre:</b>", self.styles['PatientInfo']), 
             Paragraph(paciente_data.get('nombre', 'No especificado'), self.styles['PatientInfo'])],
            [Paragraph("<b>Edad:</b>", self.styles['PatientInfo']), 
             Paragraph(f"{paciente_data.get('edad', 'N/A')} años", self.styles['PatientInfo'])],
            [Paragraph("<b>Sexo:</b>", self.styles['PatientInfo']), 
             Paragraph(paciente_data.get('sexo', 'N/A'), self.styles['PatientInfo'])],
            [Paragraph("<b>ID Paciente:</b>", self.styles['PatientInfo']), 
             Paragraph(f"ORTHO-{paciente_data.get('id', 'N/A')}", self.styles['PatientInfo'])],
            [Paragraph("<b>Fecha de Evaluación:</b>", self.styles['PatientInfo']), 
             Paragraph(paciente_data.get('fecha_creacion', datetime.datetime.now().strftime('%d/%m/%Y')), self.styles['PatientInfo'])]
        ]
        
        tabla_paciente = Table(paciente_info, colWidths=[1.5*inch, 4.5*inch])
        tabla_paciente.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F0F0')),
        ]))
        elements.append(tabla_paciente)
        return elements

    def _crear_seccion_prediccion(self, prediccion_data):
        """Crear sección de resultados de predicción"""
        elements = []
        elements.append(Paragraph("🎯 RESULTADOS DE PREDICCIÓN", self.styles['SubTitle']))
        
        prediccion_info = [
            ["Duración Estimada", f"{prediccion_data.get('prediccion', 'N/A')} meses"],
            ["Rango Probable (95%)", f"{prediccion_data.get('intervalo_min', 'N/A')} - {prediccion_data.get('intervalo_max', 'N/A')} meses"],
            ["Confianza del Modelo", f"{prediccion_data.get('confianza', 'N/A')}%"],
            ["Error Estimado (RMSE)", f"±{prediccion_data.get('error_estimado', 'N/A')} meses"],
            ["Modelo Utilizado", prediccion_data.get('modelo_usado', 'N/A').replace('_', ' ').title()]
        ]

        tabla_prediccion = Table(prediccion_info, colWidths=[2.5*inch, 3.5*inch])
        tabla_prediccion.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F0F0')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ]))
        elements.append(tabla_prediccion)
        return elements

    def _crear_graficos_plotly(self, paciente_data, prediccion_data):
        """Crear sección con gráficos Plotly exportados"""
        elements = []
        try:
            elements.append(Spacer(1, 15))
            elements.append(Paragraph("📈 ANÁLISIS VISUAL DE PREDICCIÓN", self.styles['SubTitle']))
            
            fig_prediccion = viz_system.crear_grafico_prediccion(prediccion_data, paciente_data)
            img_prediccion_base64 = viz_system.crear_grafico_para_pdf(fig_prediccion)
            
            if img_prediccion_base64:
                img_data = base64.b64decode(img_prediccion_base64)
                img_buffer = io.BytesIO(img_data)
                grafico_img = Image(img_buffer, width=6*inch, height=4*inch)
                grafico_img.hAlign = 'CENTER'
                elements.append(grafico_img)
            
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("🎯 ANÁLISIS DE COMPLEJIDAD", self.styles['SubTitle']))
            
            fig_complejidad = viz_system.crear_analisis_complejidad(paciente_data)
            img_complejidad_base64 = viz_system.crear_grafico_para_pdf(fig_complejidad)
            
            if img_complejidad_base64:
                img_complejidad_data = base64.b64decode(img_complejidad_base64)
                img_complejidad_buffer = io.BytesIO(img_complejidad_data)
                complejidad_img = Image(img_complejidad_buffer, width=6*inch, height=4.5*inch)
                complejidad_img.hAlign = 'CENTER'
                elements.append(complejidad_img)
                
        except Exception as e:
            logger.error(f"Error creando gráficos Plotly para PDF: {e}")
            elements.append(Paragraph("Error al generar gráficos. Se muestra versión simplificada.", self.styles['Highlight']))
            elements.extend(self._crear_graficos_simplificados(paciente_data, prediccion_data))
        
        return elements

    def _crear_seccion_parametros_detallados(self, paciente_data):
        """Crear sección de parámetros clínicos con análisis visual"""
        elements = []
        elements.append(Paragraph("🔬 PARÁMETROS CLÍNICOS DETALLADOS", self.styles['SubTitle']))
        
        parametros = [
            ["Parámetro Clínico", "Valor Registrado", "Interpretación"],
            ["Apiñamiento Inferior", f"{paciente_data.get('apiñamiento_mm', 'N/A')} mm", self._interpretar_apiñamiento(paciente_data.get('apiñamiento_mm', 0))],
            ["Sobremordida", f"{paciente_data.get('sobremordida_mm', 'N/A')} mm", self._interpretar_sobremordida(paciente_data.get('sobremordida_mm', 0))],
            ["Sobresalte", f"{paciente_data.get('sobresalte_mm', 'N/A')} mm", self._interpretar_sobresalte(paciente_data.get('sobresalte_mm', 0))]
        ]
        
        tabla_parametros = Table(parametros, colWidths=[2*inch, 2*inch, 2*inch])
        tabla_parametros.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(tabla_parametros)
        return elements

    def _crear_seccion_analisis_completo(self, paciente_data, prediccion_data):
        """Crear sección de análisis y recomendaciones completas"""
        elements = []
        elements.append(Paragraph("💡 ANÁLISIS Y RECOMENDACIONES", self.styles['SubTitle']))
        
        recomendaciones = self._generar_recomendaciones_completas(paciente_data)
        for i, (categoria, items) in enumerate(recomendaciones.items(), 1):
            elements.append(Paragraph(f"<b>{i}. {categoria}:</b>", self.styles['PatientInfo']))
            for item in items:
                elements.append(Paragraph(f"   • {item}", self.styles['PatientInfo']))
            elements.append(Spacer(1, 5))
        
        return elements

    def _crear_pie_pagina(self):
        """Crear pie de página profesional"""
        elements = []
        elements.append(Spacer(1, 20))
        elements.append(self._crear_linea_divisoria(colors.grey, 1))
        elements.append(Spacer(1, 10))
        
        footer_text = [
            f"{config.APP_NAME} - Sistema de Inteligencia Artificial para Ortodoncia",
            "Reporte generado automáticamente - Los resultados son predictivos y deben ser validados por un especialista.",
            f"© {datetime.datetime.now().year} {config.APP_NAME} - Todos los derechos reservados"
        ]
        
        for text in footer_text:
            elements.append(Paragraph(text, ParagraphStyle(name='Footer', fontSize=8, textColor=colors.grey, alignment=1)))
        
        return elements

    def _crear_graficos_simplificados(self, paciente_data, prediccion_data):
        """Gráficos de fallback simplificados en caso de error con Plotly"""
        elements = []
        try:
            elements.append(Paragraph("VISUALIZACIÓN DE PREDICCIÓN (Simplificada)", self.styles['SubTitle']))
            prediccion_table = [
                ["Mínimo", f"{prediccion_data.get('intervalo_min', 0)} meses", "🔴"],
                ["Predicción Principal", f"{prediccion_data.get('prediccion', 0)} meses", "🟢"], 
                ["Máximo", f"{prediccion_data.get('intervalo_max', 0)} meses", "🟠"]
            ]
            
            tabla_prediccion = Table(prediccion_table, colWidths=[2*inch, 1.5*inch, 0.5*inch])
            tabla_prediccion.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF6B6B')),
                ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#4ECDC4')),
                ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#45B7D1')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ]))
            elements.append(tabla_prediccion)
        except Exception as e:
            logger.error(f"Error en gráficos simplificados: {e}")
        
        return elements

    def _crear_linea_divisoria(self, color=colors.black, grosor=1):
        """Crear línea divisoria"""
        from reportlab.platypus.flowables import HRFlowable
        return HRFlowable(width="100%", thickness=grosor, lineCap='round', color=color, spaceBefore=1, spaceAfter=1)

    # Métodos auxiliares para análisis y clasificación
    def _calcular_nivel_complejidad(self, apiñamiento):
        if apiñamiento < 4.5: return "Muy Leve"
        elif apiñamiento < 5.5: return "Leve"
        elif apiñamiento < 6.5: return "Moderado"
        elif apiñamiento < 7.5: return "Severo"
        else: return "Muy Severo"
    
    def _interpretar_apiñamiento(self, valor):
        if valor < 5: return "Leve"
        elif valor < 7: return "Moderado"
        else: return "Severo"
    
    def _interpretar_sobremordida(self, valor):
        if valor < 2: return "Reducida"
        elif valor <= 3: return "Normal"
        else: return "Aumentada"
    
    def _interpretar_sobresalte(self, valor):
        if valor < 2: return "Reducido"
        elif valor <= 3: return "Normal"
        else: return "Aumentado"
    
    def _generar_recomendaciones_completas(self, paciente_data):
        apiñamiento = paciente_data.get('apiñamiento_mm', 6)
        
        if apiñamiento < 5:
            return {
                "Alineación": ["Alineadores transparentes son una opción viable.", "Controles de progreso cada 8-10 semanas."],
                "Seguimiento": ["Radiografías de control anuales.", "Evaluación de higiene bucal periódica."],
                "Pronóstico": ["Excelente pronóstico con alta probabilidad de éxito."]
            }
        elif apiñamiento < 7:
            return {
                "Alineación": ["Brackets convencionales o alineadores avanzados.", "Ajustes cada 4-6 semanas."],
                "Seguimiento": ["Control clínico cada 6-8 semanas.", "Radiografías panorámicas según necesidad."],
                "Pronóstico": ["Buena respuesta al tratamiento esperada con cooperación del paciente."]
            }
        else:
            return {
                "Alineación": ["Brackets convencionales recomendados.", "Evaluar necesidad de extracciones o stripping.", "Posible uso de microimplantes para anclaje."],
                "Seguimiento": ["Control estricto cada 4-6 semanas.", "Monitoreo radiográfico del movimiento radicular."],
                "Pronóstico": ["Tratamiento complejo que requiere atención especializada y seguimiento riguroso."]
            }

# Instancia global
pdf_generator = PDFReportGenerator()
