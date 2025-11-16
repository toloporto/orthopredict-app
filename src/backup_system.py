# orthopredict_app/src/backup_system.py
import os
import json
import shutil
from datetime import datetime, timedelta
import pandas as pd

class BackupSystem:
    def __init__(self):
        self.backup_dir = "backups"
        self.auto_backup_dir = "auto_backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.auto_backup_dir, exist_ok=True)
    
    def create_backup(self, db_data: dict, backup_type="manual") -> str | None:
        """Crear backup de la base de datos"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Añadir metadata al backup
            backup_data = {
                'metadata': {
                    'timestamp': timestamp,
                    'type': backup_type,
                    'version': 'OrthoPredict v5.0',
                    'created_by': 'Sistema de Backup'
                },
                'database': db_data
            }
            
            backup_dir = self.backup_dir if backup_type == "manual" else self.auto_backup_dir
            backup_file = os.path.join(backup_dir, f"{backup_type}_backup_{timestamp}.json")
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2)
            
            # Limpiar backups antiguos
            self.clean_old_backups()
            
            return backup_file
            
        except Exception as e:
            print(f"Error en backup: {e}")
            return None
    
    def clean_old_backups(self, max_backups=10, max_auto_backups=20):
        """Eliminar backups antiguos"""
        try:
            # Limpiar backups manuales
            self._clean_backup_folder(self.backup_dir, max_backups)
            # Limpiar backups automáticos
            self._clean_backup_folder(self.auto_backup_dir, max_auto_backups)
                
        except Exception as e:
            print(f"Error limpiando backups: {e}")
    
    def _clean_backup_folder(self, folder, max_files):
        """Limpiar backups en una carpeta específica"""
        backups = []
        for file in os.listdir(folder):
            if file.endswith(".json") and ("backup_" in file or "auto_backup_" in file):
                file_path = os.path.join(folder, file)
                backups.append((file_path, os.path.getctime(file_path)))
        
        # Ordenar por fecha de creación (más antiguos primero)
        backups.sort(key=lambda x: x[1])
        
        # Mantener solo el número máximo especificado
        while len(backups) > max_files:
            old_backup = backups.pop(0)
            os.remove(old_backup[0])
            print(f"🗑️ Eliminado backup antiguo: {os.path.basename(old_backup[0])}")
    
    def list_backups(self, backup_type="all"):
        """Listar todos los backups disponibles"""
        backups = []
        
        folders = []
        if backup_type in ["all", "manual"]:
            folders.append(self.backup_dir)
        if backup_type in ["all", "auto"]:
            folders.append(self.auto_backup_dir)
        
        for folder in folders:
            for file in os.listdir(folder):
                if file.endswith(".json") and ("backup_" in file or "auto_backup_" in file):
                    file_path = os.path.join(folder, file)
                    ctime = os.path.getctime(file_path)
                    date_str = datetime.fromtimestamp(ctime).strftime("%d/%m/%Y %H:%M")
                    
                    backup_type = "Manual" if "backup_" in file else "Automático"
                    
                    backups.append({
                        'file': file_path,
                        'date': date_str,
                        'timestamp': ctime,
                        'type': backup_type,
                        'size': os.path.getsize(file_path)
                    })
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    def restore_backup(self, backup_file):
        """Restaurar sistema desde backup"""
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Validar que sea un backup válido
            if 'database' in backup_data and 'pacientes' in backup_data['database']:
                # Restaurar la base de datos
                data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
                os.makedirs(data_dir, exist_ok=True)
                db_path = os.path.join(data_dir, 'pacientes_db.json')
                
                with open(db_path, 'w', encoding='utf-8') as f:
                    json.dump(backup_data['database'], f, indent=2)
                
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error restaurando backup: {e}")
            return False
    
    def get_backup_stats(self):
        """Obtener estadísticas de backups"""
        manual_backups = len([f for f in os.listdir(self.backup_dir) if f.endswith('.json')])
        auto_backups = len([f for f in os.listdir(self.auto_backup_dir) if f.endswith('.json')])
        
        total_size = 0
        for folder in [self.backup_dir, self.auto_backup_dir]:
            for file in os.listdir(folder):
                if file.endswith('.json'):
                    file_path = os.path.join(folder, file)
                    total_size += os.path.getsize(file_path)
        
        return {
            'manual_backups': manual_backups,
            'auto_backups': auto_backups,
            'total_backups': manual_backups + auto_backups,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
    
    def export_to_excel(self, pacientes: list, output_path=None) -> str | None:
        """Exportar base de datos a Excel"""
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"export_pacientes_{timestamp}.xlsx"
            
            if not pacientes:
                return None

            df = pd.DataFrame(pacientes)
            
            # Crear Excel con múltiples hojas
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Hoja principal con todos los datos
                df.to_excel(writer, sheet_name='Pacientes', index=False)
                
                # Hoja de resumen estadístico
                summary_data = {
                    'Métrica': [
                        'Total Pacientes',
                        'Edad Promedio',
                        'Duración Promedio Predicha',
                        'Apiñamiento Promedio',
                        'Casos Leves',
                        'Casos Moderados',
                        'Casos Severos'
                    ],
                    'Valor': [
                        len(pacientes),
                        round(df['edad'].mean(), 1) if 'edad' in df.columns else 'N/A',
                        round(df['duracion_predicha'].mean(), 1) if 'duracion_predicha' in df.columns else 'N/A',
                        round(df['apiñamiento_mm'].mean(), 1) if 'apiñamiento_mm' in df.columns else 'N/A',
                        len([p for p in pacientes if p.get('apiñamiento_mm', 0) < 5]),
                        len([p for p in pacientes if 5 <= p.get('apiñamiento_mm', 0) < 7]),
                        len([p for p in pacientes if p.get('apiñamiento_mm', 0) >= 7])
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Resumen', index=False)
            
            return output_path
                
        except Exception as e:
            print(f"Error exportando a Excel: {e}")
            return None

# Instancia global del sistema de backup
backup_system = BackupSystem()