import logging
import os
from datetime import datetime

# Nombre del archivo de log
LOG_NAME = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# Ruta del directorio de logs
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)  # Crear el directorio si no existe

# Ruta completa del archivo de log
LOG_FILE_PATH = os.path.join(log_dir, LOG_NAME)

# Configuración del logging
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Archivo de log
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',  # Formato del mensaje
    level=logging.INFO,  # Nivel de logging
)

# Ejecución controlada para probar
if __name__ == '__main__':
    logging.info('Logging ha comenzado')  # Mensaje de log