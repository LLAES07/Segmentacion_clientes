import sys
from src.logger import logging
def mensaje_error_detalles(error, detalles:sys):
    _, _, ejec_ = detalles.exc_info() 

    file_name = ejec_.tb_frame.f_code.co_filename


    error = 'Un error ocurrió en el archivo python {0} en la linea {1} con el error de mensaje {2} '.format(
        file_name, ejec_.tb_lineno, str(error), 
    )
    return error


class CustomExecption(Exception):
    def __init__(self, error, detalles:sys):
        super().__init__(error)
        self.error = mensaje_error_detalles(error, detalles)

    def __str__(self):
        return self.error
    

if __name__ == '__main__':

    try:
        a=1/0
    except Exception as e:
        logging.info(f'Error de division')  # Mensaje de log
        raise CustomExecption(e, sys) 
