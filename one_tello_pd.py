import scipy.constants
import rospy
from geometry_msgs.msg import TransformStamped
import socket
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import tf_conversions
from filterpy.kalman import KalmanFilter

# Configuración del filtro de Kalman
def create_kalman_filter(dt, process_noise_std, measurement_noise_std):
    """
    Crea y configura un filtro de Kalman para estimar la derivada del error.

    Parámetros:
    dt: Intervalo de tiempo.
    process_noise_std: Desviación estándar del ruido del proceso.
    measurement_noise_std: Desviación estándar del ruido de la medición.

    Retorna:
    kf: Un objeto KalmanFilter configurado.
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    # Matriz de transición de estado
    kf.F = np.array([[1, dt],
                     [0, 1]])
    
    # Matriz de observación
    kf.H = np.array([[1, 0]])
    
    # Matriz de covarianza del proceso
    kf.Q = np.array([[dt**2 * process_noise_std**2, dt * process_noise_std**2],
                     [dt * process_noise_std**2, process_noise_std**2]])

    # Matriz de covarianza de la medición
    kf.R = np.array([[measurement_noise_std**2]])
    
    # Matriz de covarianza inicial
    kf.P *= 1e4
    
    return kf

# Función para actualizar el filtro con una nueva medición
def update_kalman_filter(kf, measurement):
    """
    Actualiza el filtro de Kalman con una nueva medición.

    Parámetros:
    kf: Objeto KalmanFilter.
    measurement: Nueva medición.

    Retorna:
    Derivada estimada del error.
    """
    kf.predict()
    kf.update(measurement)
    return kf.x[1]  # Estimación de la derivada del error

class PDController:
    def __init__(self, kp, kd):
        """
        Inicializa un controlador PD.

        Parámetros:
        kp: Ganancia proporcional.
        kd: Ganancia derivativa.
        """
        self.kp = kp
        self.kd = kd
        self.last_error = [0, 0, 0, 0, 0, 0]  # Errores anteriores
        self.error = [0, 0, 0, 0, 0, 0]  # Errores actuales
        self.target = [0, 0, 1.5]  # Objetivo deseado en x, y, z
        self.torque = [0, 0, 0]  # Torques en roll, pitch y yaw
        self.thrust = 0  # Fuerza de empuje
        self.m = 0.087  # Masa del dron en kg
        self.pose = TransformStamped()  # Pose del dron
        rospy.Subscriber("/vicon/TelloMount1/TelloMount1", TransformStamped, self.set_pose)

    def set_torque(self):
        """
        Calcula los torques y el empuje basados en los errores actuales.
        """
        # Estimación de las derivadas del error en x, y y z usando Kalman
        ex_dot = update_kalman_filter(kf, self.error[0])
        ey_dot = update_kalman_filter(kf, self.error[1])
        ez_dot = update_kalman_filter(kf, self.error[2])

        # Calcular el empuje (thrust)
        uz = self.kp * self.error[2] + self.kd * ez_dot
        self.thrust = self.m * (uz + scipy.constants.g) / (
            np.cos(self.pose.transform.rotation.x) * np.cos(self.pose.transform.rotation.y)
        )

        # Calcular el torque en roll (usando error en x)
        uy = self.kp * self.error[0] + self.kd * ex_dot
        roll_d = np.arcsin(np.clip((self.m / self.thrust) * -uy, -1, 1))
        self.error[3] = roll_d - self.pose.transform.rotation.x

        # Calcular el torque en pitch (usando error en y)
        ux = self.kp * self.error[1] + self.kd * ey_dot
        pitch_d = np.arcsin(np.clip((self.m / self.thrust) * -ux / np.cos(roll_d), -1, 1))
        self.error[4] = pitch_d - self.pose.transform.rotation.y

        # Guardar torques en roll, pitch y yaw
        self.torque[0] = self.kp * self.error[3] + self.kd * (self.last_error[3] - self.error[3])
        self.torque[1] = self.kp * self.error[4] + self.kd * (self.last_error[4] - self.error[4])
        self.torque[2] = self.kp * self.error[5] + self.kd * (self.last_error[5] - self.error[5])

        # Actualizar errores anteriores
        self.last_error = self.error.copy()

    def get_torque(self, axis):
        """
        Obtiene el torque o empuje en el eje especificado.

        Parámetros:
        axis: Eje ("x", "y", "z") o "t" para empuje.

        Retorna:
        Valor del torque o empuje.
        """
        axes = {"x": 0, "y": 1, "z": 2, "t": "thrust"}
        if axis in axes:
            value = self.torque[axes[axis]] if axis != "t" else self.thrust
            return value[0] if isinstance(value, (list, np.ndarray)) else value
        return None

    def set_pose(self, data):
        """
        Actualiza la pose del dron y recalcula los errores.

        Parámetros:
        data: Nueva pose recibida.
        """
        self.pose = data

        # Obtener y normalizar el cuaternión de la rotación
        quaternion = (
            data.transform.rotation.x,
            data.transform.rotation.y,
            data.transform.rotation.z,
            data.transform.rotation.w
        )
        norm = np.sqrt(sum(q**2 for q in quaternion))
        normalized_quaternion = tuple(q / norm for q in quaternion)

        # Convertir a ángulos de Euler
        roll, pitch, yaw = tf_conversions.transformations.euler_from_quaternion(normalized_quaternion)
        self.pose.transform.rotation.x = roll
        self.pose.transform.rotation.y = pitch
        self.pose.transform.rotation.z = yaw

        # Calcular errores de posición y yaw
        self.error[0] = self.target[0] - self.pose.transform.translation.x
        self.error[1] = self.target[1] - self.pose.transform.translation.y
        self.error[2] = self.target[2] - self.pose.transform.translation.z
        self.error[5] = 0 - self.pose.transform.rotation.z

        # Recalcular torques
        self.set_torque()

def nonlinear_transform(value):
    """
    Aplica una transformación no lineal al valor de control.

    Parámetros:
    value: Valor a transformar.

    Retorna:
    Valor transformado.
    """
    if abs(value) < 5:
        return 5  # Mantiene los valores menores a 5 iguales a 5
    return value  # Mantiene los valores mayores o iguales a 5

def send_command_dron(command, ip, drone_port):
        """
        Envía un comando al dron a través del socket UDP.

        Parámetros:
        command: Comando a mandar
        ip: Dirección IP del Tello
        drone_port: Puerto del Tello (8889)
        """
        sock.sendto(command.encode('utf-8'), (ip,drone_port))

if __name__ == "__main__":
    # Inicialización de ROS
    rospy.init_node("Tello")
    
    # Variables para almacenar datos del experimento
    tim, torque_roll, torque_pitch, error_x, error_y = [], [], [], [], []
    i = 0

    # Configuración del controlador PD
    kp, kd = 20, 20
    controller = PDController(kp, kd)

    # Configuración del filtro de Kalman
    dt = 0.008  # Intervalo de tiempo
    process_noise_std = 0.1  # Ruido del proceso
    measurement_noise_std = 0.1  # Ruido de la medición
    kf = create_kalman_filter(dt, process_noise_std, measurement_noise_std)

    # Configuración del socket UDP para enviar comandos al dron
    # TelloMount1 192.168.0.148
    # TelloMount2 192.168.0.139
    drone_ip = ['192.168.0.148']
    drone_port = 8889
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)

    # Iniciamos y despegamos el Tello
    send_command_dron('command')
    time.sleep(2)
    send_command_dron('takeoff')
    time.sleep(3)
    while not rospy.is_shutdown():
        roll = -nonlinear_transform(controller.get_torque("x"))
        pitch = -nonlinear_transform(controller.get_torque("y"))
        command = f'rc {roll} {pitch} 0 0'
        print(command)
        send_command_dron(command)
        torque_roll.append(roll)
        torque_pitch.append(pitch)
        error_x.append(controller.error[0])
        error_y.append(controller.error[1])
        tim.append(i)
        i += 1

    # Mandamos el comando rc para detener el Tello y lo aterrizamos
    command = 'rc 0 0 0 0'
    send_command_dron(command)
    send_command_dron('land')
    print("Aterrizo!")
    # Cierra el socket después de usarlo
    sock.close()
    # Guardamos los datos
    df = pd.DataFrame({
    'Tiempo': tim,
    'Torque Roll': torque_roll,
    'Torque Pitch': torque_pitch,
    'Error x': error_x,
    'Error y': error_y
    })
    # Graficar cada variable en una gráfica separada
    plt.figure(figsize=(10, 8))

    # Gráfico de Torque Roll
    plt.subplot(4, 1, 1)
    sns.lineplot(data=df, x='Tiempo', y='Torque Roll')
    plt.xlabel('Tiempo')
    plt.ylabel('Torque Roll')

    # Gráfico de Torque Pitch
    plt.subplot(4, 1, 2)
    sns.lineplot(data=df, x='Tiempo', y='Torque Pitch')
    plt.xlabel('Tiempo')
    plt.ylabel('Torque Pitch')

    # Gráfico de Torque Yaw
    plt.subplot(4, 1, 3)
    sns.lineplot(data=df, x='Tiempo', y='Error x')
    plt.xlabel('Tiempo')
    plt.ylabel('Error x')

    # Gráfico de Thrust
    plt.subplot(4, 1, 4)
    sns.lineplot(data=df, x='Tiempo', y='Error y')
    plt.xlabel('Tiempo')
    plt.ylabel('Error r')

    plt.tight_layout()  # Ajusta automáticamente el diseño de los gráficos
    plt.show()

    
