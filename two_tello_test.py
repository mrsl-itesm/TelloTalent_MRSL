import rospy
from geometry_msgs.msg import TransformStamped
import socket
import time

# Inicializa el nodo de ROS
rospy.init_node("Tello")

# Configuración de direcciones IP y puerto del dron
# TelloMount1 192.168.0.148
# TelloMount2 192.168.0.139
drones_ips = ['192.168.0.148', '192.168.0.139']
drone_port = 8889

# Crea un socket UDP para enviar comandos al dron
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Configura un tiempo de espera (timeout) para la recepción de datos
sock.settimeout(5)

# Función para enviar un comando al dron
def send_command_dron(command):
    """
    Envía un comando a todos los drones listados.

    Parámetros:
    command: Comando en forma de cadena a enviar.
    """
    for ip in drones_ips:
        sock.sendto(command.encode('utf-8'), (ip, drone_port))

# Ejemplo de comandos para iniciar y despegar el dron
send_command_dron('command')
time.sleep(2)
send_command_dron('takeoff')
time.sleep(3)

# Bucle principal para enviar comandos de control
while not rospy.is_shutdown():
    # Comando de ejemplo para mover el dron en el eje Y
    command = 'rc 0 40 0 0'
    
    # Imprime el comando actual para depuración
    print(command)
    
    # Envía el comando al dron
    send_command_dron(command)
    time.sleep(0.1)  # Intervalo de tiempo entre comandos

# Comandos para aterrizar el dron y cerrar el socket al finalizar
command = 'rc 0 0 0 0'
send_command_dron(command)
send_command_dron('land')
print("Aterrizó")

# Cierra el socket después de usarlo
sock.close()
