# orthopredict_app/src/auth_system.py
import hashlib
import json
import os
import streamlit as st
from datetime import datetime, timedelta

class AuthenticationSystem:
    def __init__(self):
        self.users_file = "users.json"
        self.session_timeout = 3600  # 1 hora en segundos
        self.load_users()
    
    def load_users(self):
        """Cargar usuarios desde archivo JSON"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            else:
                # Usuarios por defecto
                self.users = {
                    "admin": {
                        "password": self.hash_password("admin123"),
                        "role": "admin",
                        "name": "Administrador Principal",
                        "email": "admin@orthopredict.com",
                        "created_at": datetime.now().isoformat()
                    },
                    "doctor": {
                        "password": self.hash_password("doctor123"), 
                        "role": "doctor",
                        "name": "Dr. Ortodoncista",
                        "email": "doctor@orthopredict.com",
                        "created_at": datetime.now().isoformat()
                    }
                }
                self.save_users()
        except Exception as e:
            print(f"Error cargando usuarios: {e}")
            self.users = {}
            self.create_default_users()
    
    def create_default_users(self):
        """Crear usuarios por defecto"""
        self.users = {
            "admin": {
                "password": self.hash_password("admin123"),
                "role": "admin",
                "name": "Administrador Principal",
                "email": "admin@orthopredict.com",
                "created_at": datetime.now().isoformat()
            },
            "doctor": {
                "password": self.hash_password("doctor123"), 
                "role": "doctor",
                "name": "Dr. Ortodoncista",
                "email": "doctor@orthopredict.com",
                "created_at": datetime.now().isoformat()
            }
        }
        self.save_users()
    
    def save_users(self):
        """Guardar usuarios en archivo JSON"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            print(f"Error guardando usuarios: {e}")
    
    def hash_password(self, password):
        """Hashear contraseña con salt"""
        salt = "orthopredict_salt_2024"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def authenticate(self, username, password):
        """Autenticar usuario"""
        if username in self.users:
            hashed_password = self.hash_password(password)
            if self.users[username]['password'] == hashed_password:
                return True
        return False
    
    def create_user(self, username, password, role="doctor", name="", email=""):
        """Crear nuevo usuario"""
        if username not in self.users:
            self.users[username] = {
                "password": self.hash_password(password),
                "role": role,
                "name": name or username,
                "email": email or f"{username}@orthopredict.com",
                "created_at": datetime.now().isoformat()
            }
            self.save_users()
            return True
        return False
    
    def update_user(self, username, **kwargs):
        """Actualizar información de usuario"""
        if username in self.users:
            for key, value in kwargs.items():
                if key != 'password' and value:
                    self.users[username][key] = value
            self.save_users()
            return True
        return False
    
    def change_password(self, username, new_password):
        """Cambiar contraseña de usuario"""
        if username in self.users:
            self.users[username]['password'] = self.hash_password(new_password)
            self.save_users()
            return True
        return False
    
    def delete_user(self, username):
        """Eliminar usuario (solo admin)"""
        if username in self.users and username != 'admin':  # Proteger admin
            del self.users[username]
            self.save_users()
            return True
        return False
    
    def get_user_role(self, username):
        """Obtener rol de usuario"""
        return self.users.get(username, {}).get('role', 'guest')
    
    def get_all_users(self):
        """Obtener todos los usuarios"""
        return self.users

# Instancia global del sistema de autenticación
auth_system = AuthenticationSystem()

def check_authentication():
    """Verificar si el usuario está autenticado"""
    # Inicializar session_state si no existe
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.name = None
        st.session_state.login_time = None
    
    # Verificar timeout de sesión
    if (st.session_state.authenticated and st.session_state.login_time and 
        (datetime.now() - st.session_state.login_time).seconds > auth_system.session_timeout):
        logout()
        st.warning("🕒 Sesión expirada por inactividad")
        return False
    
    return st.session_state.authenticated

def login(username, password):
    """Iniciar sesión"""
    if auth_system.authenticate(username, password):
        st.session_state.authenticated = True
        st.session_state.user = username
        st.session_state.role = auth_system.get_user_role(username)
        st.session_state.name = auth_system.users[username]['name']
        st.session_state.login_time = datetime.now()
        return True
    return False

def logout():
    """Cerrar sesión"""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.name = None
    st.session_state.login_time = None

def require_auth():
    """Decorador para requerir autenticación"""
    if not check_authentication():
        st.error("🔒 Acceso denegado. Debes iniciar sesión.")
        st.stop()

def require_role(required_role):
    """Decorador para requerir rol específico"""
    require_auth()
    if st.session_state.role != required_role:
        st.error(f"🔒 Se requiere rol de {required_role} para acceder a esta función.")
        st.stop()

def login_page():
    """Página de login"""
    st.title("🔐 OrthoPredict Pro ML - Login")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.subheader("Iniciar Sesión")
            username = st.text_input("Usuario", placeholder="Ingresa tu usuario")
            password = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
            
            if st.form_submit_button("🚀 Ingresar al Sistema", use_container_width=True):
                if login(username, password):
                    st.success(f"¡Bienvenido, {st.session_state.name}! 👋")
                    st.rerun()
                else:
                    st.error("❌ Usuario o contraseña incorrectos")
        
        # Información de usuarios demo
        with st.expander("ℹ️ Usuarios de Demo", expanded=True):
            st.write("**👑 Administrador:**")
            st.code("Usuario: admin\nContraseña: admin123")
            st.write("**👨‍⚕️ Doctor:**")
            st.code("Usuario: doctor\nContraseña: doctor123")
            
        st.info("💡 **Sistema de autenticación seguro** - Las contraseñas están hasheadas y protegidas")