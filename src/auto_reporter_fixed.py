# orthopredict_app/src/auto_reporter_fixed.py
import pandas as pd
import json
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText  # ✅ CORREGIDO
from email.mime.multipart import MIMEMultipart  # ✅ CORREGIDO
from email.mime.base import MIMEBase
from email import encoders
import os
import logging
from typing import List, Dict, Optional

from monitoring_system import MLMonitoringSystem
from config import config

logger = logging.getLogger(__name__)

class AutoReporter:
    """Sistema de reportes automáticos para OrthoPredict"""
    
    def __init__(self, monitoring_system: MLMonitoringSystem):
        self.monitoring_system = monitoring_system
        self.report_config = {
            'daily_report_time': '08:00',  # Hora para reporte diario
            'weekly_report_day': 'monday',  # Día para reporte semanal
            'alert_threshold': 5,  # Número de alertas para notificación
        }
    
    def generate_daily_report(self) -> Dict:
        """Generar reporte diario automático"""
        logger.info("Generando reporte diario automático")
        
        reporte = self.monitoring_system.generar_reporte_monitoreo(1)  # Últimas 24 horas
        
        report_data = {
            'tipo': 'diario',
            'fecha_generacion': datetime.now().isoformat(),
            'periodo': '24 horas',
            'resumen': reporte.get('resumen', {}),
            'alertas': reporte.get('alertas_recientes', {}),
            'recomendaciones': reporte.get('recomendaciones', []),
            'cache_stats': reporte.get('metricas_cache', {})
        }
        
        # Guardar reporte
        self._save_report(report_data, 'daily')
        
        return report_data
    
    def generate_weekly_report(self) -> Dict:
        """Generar reporte semanal automático"""
        logger.info("Generando reporte semanal automático")
        
        reporte = self.monitoring_system.generar_reporte_monitoreo(7)  # Última semana
        
        report_data = {
            'tipo': 'semanal',
            'fecha_generacion': datetime.now().isoformat(),
            'periodo': '7 días',
            'resumen': reporte.get('resumen', {}),
            'tendencias': reporte.get('tendencias', {}),
            'alertas': reporte.get('alertas_recientes', {}),
            'recomendaciones': reporte.get('recomendaciones', []),
            'metricas_detalladas': self._get_detailed_metrics(7)
        }
        
        # Guardar reporte
        self._save_report(report_data, 'weekly')
        
        return report_data
    
    def check_and_send_alerts(self) -> bool:
        """Verificar y enviar alertas si es necesario"""
        alertas_recientes = self.monitoring_system.alertas
        
        # Filtrar alertas de las últimas 24 horas
        recent_alerts = [
            alert for alert in alertas_recientes
            if datetime.now() - datetime.fromisoformat(alert['fecha']) < timedelta(hours=24)
        ]
        
        critical_alerts = [a for a in recent_alerts if a['nivel'] == 'CRITICO']
        
        if critical_alerts or len(recent_alerts) >= self.report_config['alert_threshold']:
            logger.warning(f"Enviando notificación de alertas: {len(critical_alerts)} críticas, {len(recent_alerts)} totales")
            return self._send_alert_notification(critical_alerts, recent_alerts)
        
        return False
    
    def _save_report(self, report_data: Dict, report_type: str):
        """Guardar reporte en archivo"""
        try:
            reports_dir = os.path.join(config.DATA_DIR, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{report_type}_report_{timestamp}.json"
            filepath = os.path.join(reports_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Reporte guardado: {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando reporte: {e}")
    
    def _get_detailed_metrics(self, days: int) -> Dict:
        """Obtener métricas detalladas para el período"""
        metrics_data = []
        
        for eval_data in self.monitoring_system.metricas_historicas:
            eval_date = datetime.fromisoformat(eval_data['fecha'])
            if datetime.now() - eval_date < timedelta(days=days):
                if 'metricas_performance' in eval_data:
                    metrics_data.append(eval_data['metricas_performance'])
        
        if not metrics_data:
            return {}
        
        df = pd.DataFrame(metrics_data)
        
        return {
            'r2_stats': {
                'mean': df['r2'].mean(),
                'std': df['r2'].std(),
                'min': df['r2'].min(),
                'max': df['r2'].max()
            },
            'mae_stats': {
                'mean': df['mae'].mean(),
                'std': df['mae'].std(),
                'min': df['mae'].min(),
                'max': df['mae'].max()
            }
        }
    
    def _send_alert_notification(self, critical_alerts: List, all_alerts: List) -> bool:
        """Enviar notificación de alertas por email"""
        try:
            # Esta función necesitaría configuración SMTP real
            # Por ahora solo loggueamos la intención
            logger.info(f"SIMULACIÓN: Enviando notificación de {len(critical_alerts)} alertas críticas y {len(all_alerts)} alertas totales")
            
            # En una implementación real, aquí iría el código SMTP
            # para enviar emails con las alertas
            
            return True
            
        except Exception as e:
            logger.error(f"Error enviando notificación: {e}")
            return False

# Sistema de scheduling para reportes automáticos
class ReportScheduler:
    """Programador de reportes automáticos"""
    
    def __init__(self, auto_reporter: AutoReporter):
        self.auto_reporter = auto_reporter
        self.last_daily_report = None
        self.last_weekly_report = None
    
    def check_scheduled_reports(self):
        """Verificar y ejecutar reportes programados"""
        now = datetime.now()
        
        # Reporte diario a las 8:00 AM
        if now.hour == 8 and now.minute == 0:
            if self.last_daily_report != now.date():
                self.auto_reporter.generate_daily_report()
                self.last_daily_report = now.date()
                logger.info("Reporte diario automático generado")
        
        # Reporte semanal los lunes a las 9:00 AM
        if now.weekday() == 0 and now.hour == 9 and now.minute == 0:  # Lunes
            if not self.last_weekly_report or (now - self.last_weekly_report).days >= 7:
                self.auto_reporter.generate_weekly_report()
                self.last_weekly_report = now
                logger.info("Reporte semanal automático generado")
        
        # Verificar alertas cada hora
        if now.minute == 0:  # Cada hora en punto
            self.auto_reporter.check_and_send_alerts()