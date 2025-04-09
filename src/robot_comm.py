# Description: This module is responsible for communication with the robot via Modbus TCP protocol.
# It uses the pyModbusTCP library to establish a connection with the robot and send data to it.
# robot_comm.py
#pip install pyModbusTCP 
      
import logging # Ensure logging is imported if not already
from datetime import datetime # Ensure datetime is imported
from pyModbusTCP.client import ModbusClient
import os

class RobotComm:
    def __init__(self, host="192.168.0.10", port=502, timeout=5, log_file='app_data/error_log.txt'):
        """
        :param host: IP address (str)
        :param port: Modbus TCP port (int, default: 502)
        :param timeout: Connection timeout in seconds (int)
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.log_file = log_file
        self.client = ModbusClient(host=self.host, port=self.port, timeout=self.timeout)
        self.connected = False

        
        if not os.path.exists(os.path.dirname(self.log_file)):
            os.makedirs(os.path.dirname(self.log_file))
        logging.basicConfig(filename=self.log_file,
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def connect(self):
        try:
            self.connected = self.client.open()
            if self.connected:
                logging.info("Connected.")
                logging.info(f"Connected to Modbus server at {self.host}:{self.port}")
            else:
                logging.error("Failed.")
                logging.error(f"Failed to connect to Modbus server at {self.host}:{self.port}")
        except Exception as e:
            self.connected = False
            logging.error(f"Connection error: {str(e)}")
            logging.error(f"Exception during connection: {e}")

    def disconnect(self):
        if self.connected:
            self.client.close()
            self.connected = False
            logging.info("Disconnected")
            logging.info(f"Disconnected from Modbus server.")
    def read_request_flag(self, address, is_coil=False):
        """
        Reads a single register or coil to check for a data request flag.
        Args:
            address: Address of the flag register/coil.
            is_coil: Set to True if reading a coil, False for Holding Register.
        Returns:
            The value read (e.g., 0 or 1), or None on error.
        """
        if not self.connected:
            logging.error("Cannot read flag: Not connected.")
            # print(f"[{datetime.now()}] Not connected. Cannot read flag.") # Keep logging primary
            return None

        try:
            if is_coil:
                # Read Coils (function code 1)
                result = self.client.read_coils(address, 1)
            else:
                # Read Holding Registers (function code 3)
                result = self.client.read_holding_registers(address, 1)

            if result: # Check if result is not None or empty
                # logging.info(f"Read flag at address {address}: {result[0]}") # Log only on change or request?
                return result[0] # Return the first (and only) value
            else:
                last_ex = self.client.last_exception()
                last_ex_str = str(last_ex) if last_ex else "Unknown read error"
                logging.error(f"Failed to read flag at address {address}. Error: {last_ex_str}")
                # print(f"[{datetime.now()}] Failed to read flag at address {address}. Error: {last_ex_str}")
                return None
        except Exception as e:
            logging.error(f"Exception during flag reading at address {address}: {str(e)}")
            # print(f"[{datetime.now()}] Exception during reading flag: {e}")
            return None

    def reset_request_flag(self, address, is_coil=False):
        """
        Resets the request flag by writing 0 to it.
        Args:
            address: Address of the flag register/coil.
            is_coil: Set to True if writing to a coil.
        Returns:
            True on success, False on failure.
        """
        if not self.connected:
            logging.error("Cannot reset flag: Not connected.")
            return False

        try:
            if is_coil:
                # Write Single Coil (function code 5)
                success = self.client.write_single_coil(address, False) # False means OFF (0)
            else:
                # Write Single Register (function code 6)
                success = self.client.write_single_register(address, 0)

            if success:
                logging.info(f"Successfully reset flag at address {address}")
            else:
                logging.error(f"Failed to reset flag at address {address}")
            return success
        except Exception as e:
            logging.error(f"Exception during flag reset at address {address}: {str(e)}")
            return False



def send_data(self, obj_id, x_mm, y_mm, width_mm, height_mm, angle, category_code):
    """
    Sends detected object data to the robot via Modbus registers.

    :param obj_id: Unique identifier for the object (int). Must be a positive integer.
    :param x_mm: X-coordinate of object in mm (float). Must be within the range [0, 10000].
    :param y_mm: Y-coordinate of object in mm (float). Must be within the range [0, 10000].
    :param width_mm: Width of object in mm (float). Must be within the range [0, 5000].
    :param height_mm: Height of object in mm (float). Must be within the range [0, 5000].
    :param angle: Angle of rotation of object (float). Must be within the range [0, 360].
    :param category_code: Numeric category code of the object (int). Must be a positive integer within a predefined set of valid codes.
    """

    if not self.connected:
        logging.error("Attempted to send data without connection.")
        logging.error("Not connected. Cannot send data.")
        return False

    logging.info("Attempting to reconnect...")
    self.connect()
    if not self.connected:
        logging.error("Reconnection failed. Cannot send data.")
        logging.error("Reconnection failed. Cannot send data.")
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
        success = self.client.write_registers(0, data)
        if success:
            logging.info(f"Sent data successfully: {data}")
            logging.info(f"Data sent: {data}")
        else:
            logging.error("Failed to send.")
            logging.error(f"Failed to send data: {data}")
        return success
    except Exception as e:
        logging.error(f"Exception during data sending: {str(e)}")
        logging.error(f"Exception during sending data: {e}")
        return False
