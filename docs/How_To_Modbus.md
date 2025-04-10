# PLC Setup Guide for Modbus TCP Communication with Python Vision App

**Objective:**  
Configure a PLC to request object data from a Python application and receive multiple detected object packets via Modbus TCP.

## Assumptions

- Your PLC supports Modbus TCP Server.
- You have PLC programming software.
- The PLC and Python PC are connected within the same Ethernet network (IP, subnet mask, gateway, firewall properly configured).

## Key Parameters (Configured via Python GUI)

Before configuring your PLC, open the Python application, navigate to the "Modbus TCP" tab, and verify or record these parameters:

- **IP Address:** Static IP assigned to PLC (e.g. `192.168.0.10`)
- **Port:** Modbus TCP port (usually `502`)
- **Request Flag Addr:** Modbus address for the data request flag (e.g. `100`)
- **Flag is Coil (checkbox):** Checked (`True`) if the flag is a Coil, unchecked (`False`) if it's a Holding Register
- **Num Objects Addr:** Modbus address holding the number of objects (`200`)
- **Object Data Start Addr:** Starting Modbus address for the first object's data (`201`)
- **Max Objects/Packet:** Maximum number of objects PLC can handle at once (e.g. `5`)

**Note:** The parameter `Registers/Object` (`MODBUS_REGISTERS_PER_OBJECT`) is usually fixed in Python code (value `7`). Ensure this matches your PLC data structure.

## Step-by-Step PLC Configuration

### Step 1: Network and Modbus TCP Server Configuration

- **Assign IP Address:**
  - In PLC hardware or network settings, set the static IP address matching Python GUI settings.
  - Confirm subnet mask and gateway are properly configured.

- **Enable Modbus TCP Server:**
  - Locate the Modbus TCP Server settings in PLC software.
  - Enable the Modbus TCP Server function.

- **Set the Port:**
  - Ensure Modbus TCP Server port matches the port set in Python GUI (typically `502`).

- **Configure Access (if required):**
  - Some PLCs require IP whitelisting. If needed, add Python PC IP to allowed clients.
  - Ensure Read/Write access is permitted for Holding Registers/Coils used by Python.

### Step 2: Creating PLC Tags/Variables

Create PLC tags (variables) matching the Modbus addresses specified in the Python GUI. Syntax may vary depending on your PLC software.

**Request Flag Tag:**

- Name: `Vision_DataRequestFlag`
- Data Type:
  - `BOOL` if "Flag is Coil" is checked.
  - `INT`, `UINT`, or `WORD` if unchecked.
- Modbus Address: as set in `Request Flag Addr`

**Number of Objects Tag:**

- Name: `Vision_NumObjectsReceived`
- Data Type: `INT`, `UINT`, or `WORD` (16-bit)
- Modbus Address: as set in `Num Objects Addr`

**Object Data Array:**

- **Option 1 (Preferred): UDT (Structure)**
  
Define a user-defined data type (`UDT`):

```plaintext
TYPE UDT_VisionObject :
STRUCT
    TrackID         : INT;
    CenterX_x100    : INT;
    CenterY_x100    : INT;
    Width_x100      : INT;
    Height_x100     : INT;
    Angle_x100      : INT;
    CategoryCode    : INT;
END_STRUCT
END_TYPE
```

Ensure the order and number of fields exactly match the Python data structure.

Create an array of this UDT:

- **Name:** `Vision_ObjectDataArray`
- **Type:** `ARRAY[0..N-1] OF UDT_VisionObject` *(N = Max Objects/Packet from GUI)*
- **Modbus Address:** Assign the start address (from Object Data Start Addr) to the first element's first field (`Vision_ObjectDataArray[0].TrackID`).

### Option 2 (Less Preferred): Simple INT/WORD Array

Create a simple integer array:

- **Name:** `Vision_RawObjectData`
- **Type:** `ARRAY[0..M-1] OF INT` *(M = Max Objects/Packet × 7)*
- **Modbus Address:** Assign the start address (from Object Data Start Addr) to the first element (`Vision_RawObjectData[0]`).

*(Note: Less convenient due to manual indexing.)*

## Step 3: Implementing PLC Logic

Write logic (Ladder Logic, Structured Text, FBD, etc.) to control data communication:

### Data Request:
- **Condition:** PLC ready (robot is free) AND previous processing done (`YourProcessingDoneFlag = TRUE`) AND `Vision_DataRequestFlag = 0`.
- Set `Vision_DataRequestFlag = 1`.
- Reset internal readiness flag (`YourProcessingDoneFlag = FALSE`).

### Waiting and Data Reception:
- Python app detects `Vision_DataRequestFlag = 1`, sends object data, writes to `Vision_NumObjectsReceived` and array, then resets flag to `0`.
- PLC detects reset (`Vision_DataRequestFlag` changed from `1` to `0`).
- Read number of objects (`Vision_NumObjectsReceived`) into a local variable `N`.
- **Validate:** Ensure `0 <= N <= Max Objects/Packet`. If invalid, set `N = 0`.

### Processing Received Data:
- Loop from `0` to `N-1`:
  - Access each object's data (`Vision_ObjectDataArray[i]` or calculated indices).
  - Convert values: Divide `_x100` fields by `100.0` to obtain real-world values.
  - Use ID, CategoryCode, and converted values (coordinates, size, angle) for robot control, sorting, visualization, etc.

### Completion of Processing:
- After loop completion (or if `N = 0`), set internal readiness flag (`YourProcessingDoneFlag = TRUE`).
- Optionally verify `Vision_DataRequestFlag` is reset to `0`. Reset manually if necessary.

## Step 4: Testing and Debugging

### Modbus Client Tool:
- Use a third-party Modbus TCP client to verify PLC communication.
- Test reading/writing tags (`Vision_DataRequestFlag`, object data).

### PLC Software Monitoring:
- Monitor tag values in real-time during Python application execution.

### Logs:
- Check Python logs (`error_log.txt`, console output) and PLC built-in logging tools.