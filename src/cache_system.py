# orthopredict_app/src/cache_system.py
import time
from typing import Any, Optional, Dict, Tuple
import hashlib
import json
from cachetools import TTLCache, LRUCache
import logging
from config import config

logger = logging.getLogger(__name__)

class OrthoCacheSystem:
    """Sistema de cache optimizado para OrthoPredict"""
    
    def __init__(self):
        self.enabled = config.CACHE_ENABLED
        self.caches: Dict[str, TTLCache] = {}
        
        # Inicializar caches específicos
        self._init_caches()
        
        # Estadísticas
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
    
    def _init_caches(self):
        """Inicializar caches específicos con diferentes configuraciones"""
        cache_configs = {
            'predictions': {
                'maxsize': 200,  # Cache de 200 predicciones
                'ttl': 1800      # 30 minutos
            },
            'models': {
                'maxsize': 10,   # Cache de 10 modelos
                'ttl': 3600      # 1 hora
            },
            'reports': {
                'maxsize': 50,   # Cache de 50 reportes
                'ttl': 900       # 15 minutos
            },
            'visualizations': {
                'maxsize': 100,  # Cache de 100 visualizaciones
                'ttl': 1800      # 30 minutos
            },
            'patient_data': {
                'maxsize': 500,  # Cache de 500 pacientes
                'ttl': 3600      # 1 hora
            }
        }
        
        for cache_name, cache_config in cache_configs.items():
            self.caches[cache_name] = TTLCache(
                maxsize=cache_config['maxsize'],
                ttl=cache_config['ttl']
            )
    
    def _generate_key(self, cache_name: str, *args, **kwargs) -> str:
        """Generar clave única para cache basada en parámetros"""
        key_data = f"{cache_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, cache_name: str, *args, **kwargs) -> Optional[Any]:
        """Obtener elemento del cache"""
        if not self.enabled:
            return None
        
        if cache_name not in self.caches:
            logger.warning(f"Cache '{cache_name}' no encontrado")
            return None
        
        key = self._generate_key(cache_name, *args, **kwargs)
        
        try:
            value = self.caches[cache_name].get(key)
            if value is not None:
                self.stats['hits'] += 1
                logger.debug(f"Cache HIT: {cache_name} - {key[:8]}")
                return value
            else:
                self.stats['misses'] += 1
                logger.debug(f"Cache MISS: {cache_name} - {key[:8]}")
                return None
        except Exception as e:
            logger.error(f"Error accediendo al cache {cache_name}: {e}")
            return None
    
    def set(self, cache_name: str, value: Any, *args, **kwargs) -> bool:
        """Guardar elemento en cache"""
        if not self.enabled:
            return False
        
        if cache_name not in self.caches:
            logger.warning(f"Cache '{cache_name}' no encontrado")
            return False
        
        key = self._generate_key(cache_name, *args, **kwargs)
        
        try:
            # Verificar si hay evicción
            current_size = len(self.caches[cache_name])
            if key not in self.caches[cache_name] and current_size >= self.caches[cache_name].maxsize:
                self.stats['evictions'] += 1
            
            self.caches[cache_name][key] = value
            self.stats['sets'] += 1
            logger.debug(f"Cache SET: {cache_name} - {key[:8]}")
            return True
        except Exception as e:
            logger.error(f"Error guardando en cache {cache_name}: {e}")
            return False
    
    def clear(self, cache_name: Optional[str] = None):
        """Limpiar cache específico o todos los caches"""
        try:
            if cache_name:
                if cache_name in self.caches:
                    self.caches[cache_name].clear()
                    logger.info(f"Cache '{cache_name}' limpiado")
                else:
                    logger.warning(f"Cache '{cache_name}' no encontrado para limpiar")
            else:
                for name, cache in self.caches.items():
                    cache.clear()
                logger.info("Todos los caches limpiados")
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}")
    
    def invalidate_pattern(self, cache_name: str, pattern: str):
        """Invalidar entradas de cache que coincidan con un patrón"""
        if cache_name not in self.caches:
            return
        
        try:
            keys_to_remove = [
                key for key in self.caches[cache_name].keys()
                if pattern in key
            ]
            
            for key in keys_to_remove:
                del self.caches[cache_name][key]
            
            logger.info(f"Invalidadas {len(keys_to_remove)} entradas en cache '{cache_name}' con patrón '{pattern}'")
        except Exception as e:
            logger.error(f"Error invalidando cache por patrón: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        total_operations = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_operations * 100) if total_operations > 0 else 0
        
        cache_sizes = {}
        for cache_name, cache in self.caches.items():
            cache_sizes[cache_name] = {
                'current_size': len(cache),
                'max_size': cache.maxsize,
                'usage_percentage': (len(cache) / cache.maxsize * 100) if cache.maxsize > 0 else 0
            }
        
        return {
            'enabled': self.enabled,
            'hit_rate': round(hit_rate, 2),
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'total_sets': self.stats['sets'],
            'total_evictions': self.stats['evictions'],
            'cache_sizes': cache_sizes
        }
    
    def prefetch_patient_data(self, patient_ids: list):
        """Pre-cargar datos de pacientes en cache"""
        if not self.enabled:
            return
        
        try:
            # Esta función se integrará con el sistema de base de datos
            # Por ahora es un placeholder para la funcionalidad
            logger.info(f"Pre-cargando datos para {len(patient_ids)} pacientes")
        except Exception as e:
            logger.error(f"Error en pre-carga de datos: {e}")

# Instancia global del sistema de cache
cache_system = OrthoCacheSystem()