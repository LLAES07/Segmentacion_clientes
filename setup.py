# Responsable de generar mi machine learning como un paquete
from setuptools import find_packages, setup
import os


HYPEN_E = '-e .'
def get_paquetes(filepath:str)->list[str]:
    
    items = list()
    with open(filepath, 'r') as f:

        '''
        Función para retornar la lista de paquetes
        '''
        items = [line.replace('\n', '') for line in f.readlines() ]
        if HYPEN_E in items:
            # SI el término está presente en la lista entonces se remueve.
            items.remove(HYPEN_E)
    return items
    
setup(
    name='ml_project',
    version='0.0.1',
    author='Kevin Llanos',
    author_email='llanos.k.geology@gmail.com',
    packages=find_packages(),
    install_requires=get_paquetes('requirements.txt')
)

if __name__=='main':
    ahora = get_paquetes('requirements.txt')
    print(type(ahora))