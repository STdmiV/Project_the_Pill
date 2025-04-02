# Description: This module is responsible for communication with the robot via Modbus TCP protocol.
# It uses the pyModbusTCP library to establish a connection with the robot and send data to it.
# robot_comm.py
#pip install pyModbusTCP 
# instal that library




from pyModbusTCP.client import ModbusClient
import logging
from datetime import datetime

class RobotComm:
    def __init__(self, host="192.168.0.10", port=502, timeout=5):
        """
        :param host: IP address
        :param port: Modbus TCP port (default: 502)
        :param timeout: Connection timeout 's'
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = ModbusClient(host=self.host, port=self.port, timeout=self.timeout)
        self.connected = False

        logging.basicConfig(filename='app_data/error_log.txt',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def connect(self):
        try:
            self.connected = self.client.open()
            if self.connected:
                logging.info("Connected.")
                print(f"[{datetime.now()}] Connected to Modbus server at {self.host}:{self.port}")
            else:
                logging.error("Failed.")
                print(f"[{datetime.now()}] Failed to connect.")
        except Exception as e:
            self.connected = False
            logging.error(f"Connection error: {str(e)}")
            print(f"[{datetime.now()}] Exception during connection: {e}")

    def disconnect(self):
        if self.connected:
            self.client.close()
            self.connected = False
            logging.info("Disconnected")
            print(f"[{datetime.now()}] Disconnected from Modbus server.")




########################################################################
        """
        Sends detected object data to the robot via Modbus registers.

        :param obj_id: Unique identifier for the object (int)
        :param x_mm: X-coordinate of object in mm (float)
        :param y_mm: Y-coordinate of object in mm (float)
        :param width_mm: Width of object in mm (float)
        :param height_mm: Height of object in mm (float)
        :param angle: Angle of rotation of object (float)
        :param category_code: Numeric category code of the object (int)
        """
    
    def send_data(self, obj_id, x_mm, y_mm, width_mm, height_mm, angle, category_code):

        if not self.connected:
            logging.error("send data without connection.")
            print(f"[{datetime.now()}] Not connected. Cannot send data.")
            return False

        data = [
            obj_id,
            int(x_mm * 100),
            int(y_mm * 100),
            int(width_mm * 100),
            int(height_mm * 100),
            int(angle * 100),
            category_code
        ]
        try:
            # starting from register address 0
            success = self.client.write_multiple_registers(0, data)
            if success:
                logging.info(f"Sent data successfully: {data}")
                print(f"[{datetime.now()}] Data sent: {data}")
            else:
                logging.error("Failed to send.")
                print(f"[{datetime.now()}] Failed to send data: {data}")
            return success
        except Exception as e:
            logging.error(f"Exception during data sending: {str(e)}")
            print(f"[{datetime.now()}] Exception during sending data: {e}")
            return False
