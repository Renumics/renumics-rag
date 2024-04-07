from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from assistant.types import LLM

PROMPT_TEMPLATE = """### Instructions:
Your task is to convert a question into a SQL query, given the SQL table schema.

SQL dataset contains a racing team dataset. It represents a racing car training session. The dataset contains timestamps at constant 1000 Hz frequency and multiple CAN-bus signals at these timestamps.

Don't forget to query _datetime and _timestamp columns since they are crucial for the data understanding.

### Input:
Generate a SQL query that answers the question `{question}`.
This query will run on a database whose schema is represented in this string:
CREATE TABLE dataset (
    _datetime TIMESTAMP_NS, -- Absolute datetime.
    _timestamp DOUBLE, -- Time passed since the Unix epoch in seconds.
    a1_iu_gps_power DOUBLE, -- Power supply status of the GPS on the Intelligent Unit (IU).
    a1_pitorohr_current DOUBLE, -- Current flowing through the pitot tube's heater, which is used to measure airspeed.
    a2_com_router_power DOUBLE, -- Power supply status of the communication router responsible for network connectivity.
    a2_le_rear_ign_current DOUBLE, -- Current flowing to the Left Engine's rear ignition system.
    a4_bibf_pitot_kmdf_rts_power DOUBLE, -- Power status for a specific pitot tube system (possibly with abbreviations BIBF, KMDF, RTS referring to components or system names).
    a4_iu_bibs_front_imu_rt_current DOUBLE, -- Current to the front Inertial Measurement Unit (IMU) on the Intelligent Unit, possibly in real-time (RT).
    brake_pedal_position_1 DOUBLE, -- Position of the first brake pedal sensor, indicating the pedal's deflection.
    brake_pedal_position_2 DOUBLE, -- Position of the second brake pedal sensor, indicating the pedal's deflection.
    brake_percent_1 DOUBLE, -- Braking force as a percentage applied via the first brake sensor.
    brake_percent_2 DOUBLE, -- Braking force as a percentage applied via the second brake sensor.
    brake_pressure_front DOUBLE, -- Hydraulic brake pressure in the front brake circuit.
    current DOUBLE, -- General current measurement, further details needed to give a specific meaning.
    f1_shutdown_circuit_power DOUBLE, -- Power status of the shutdown circuit specific to a car component or system, possibly a safety feature.
    g1_mcu_power DOUBLE, -- Power supply to the Main Control Unit (MCU).
    gearbox_oil_temp_fl DOUBLE, -- Temperature of the oil in the gearbox at the front-left (FL) position.
    gps_position_lat DOUBLE, -- GPS-determined latitude.
    gps_position_long DOUBLE, -- GPS-determined longitude.
    gps_time_day DOUBLE, -- Day of the month as given by GPS time.
    gps_time_hour DOUBLE, -- Hour of the day as given by GPS time.
    gps_time_minute DOUBLE, -- Minute of the hour as given by GPS time.
    gps_time_month DOUBLE, -- Month of the year as given by GPS time.
    gps_time_second DOUBLE, -- Second of the minute as given by GPS time.
    gps_time_year DOUBLE, -- Year as given by GPS time.
    j1_logger_ethnetswitchf_power DOUBLE, -- Power status of the Ethernet network switch connected to a logger, perhaps for data acquisition.
    l1_drs_current DOUBLE, -- Current associated with the Drag Reduction System (DRS), a system used to reduce aerodynamic drag.
    l2_vcc_le_front_current DOUBLE, -- Current to a specific component in the Left Engine or Vehicle Control Computer at the front.
    l3_motor_fan_power_pwm_current DOUBLE, -- Current to the motor fan being controlled by Pulse Width Modulation (PWM) for speed variability.
    l4_pe_pump_current DOUBLE, -- Current to a pump, possibly related to power electronics cooling or hydraulic systems.
    lvbms_chargefetstatus DOUBLE, -- Status of the charge Field-Effect Transistor (FET) in the Low Voltage Battery Management System.
    lvbms_current_afe DOUBLE, -- Current measurement from the Analog Front-End (AFE) in the Low Voltage Battery Management System.
    lvbms_dischargefetstatus DOUBLE, -- Status of the discharge Field-Effect Transistor (FET) in the Low Voltage Battery Management System.
    lvbms_nowarningsandfaults DOUBLE, -- Indicator if there are no warnings or faults in the Low Voltage Battery Management System.
    lvbms_temp0_cell DOUBLE, -- Temperature of the first cell in the Low Voltage Battery Management System.
    lvbms_temp1_cell DOUBLE, -- Temperature of the second cell in the Low Voltage Battery Management System.
    lvbms_temp2_cell DOUBLE, -- Temperature of the third cell in the Low Voltage Battery Management System.
    lvbms_temp_fet0 DOUBLE, -- Temperature of the first Field-Effect Transistor in the Low Voltage Battery Management System.
    lvbms_temp_fet1 DOUBLE, -- Temperature of the second Field-Effect Transistor in the Low Voltage Battery Management System.
    lvbms_voltage_afe DOUBLE, -- Voltage measurement from the Analog Front-End in the Low Voltage Battery Management System.
    lvbms_voltage_cell_1_0 DOUBLE, -- Voltage of the first cell in the first bank/module of the Low Voltage Battery Management System.
    lvbms_voltage_cell_1_1 DOUBLE, -- Voltage of the second cell in the first bank/module of the Low Voltage Battery Management System.
    lvbms_voltage_cell_1_2 DOUBLE, -- Voltage of the third cell in the first bank/module of the Low Voltage Battery Management System.
    lvbms_voltage_cell_1_3 DOUBLE, -- Voltage of the fourth cell in the first bank/module of the Low Voltage Battery Management System.
    m1_cool_pump_motor_current DOUBLE, -- Current to the cooling pump motor, likely for the engine or power electronics.
    m2_vcc_le_rear_current DOUBLE, -- Current to a component or system in the vehicle control computer or Left Engine at the rear position.
    m3_hv_fan_current DOUBLE, -- Current to the High Voltage system fan, used for cooling.
    m4_le_fan_current DOUBLE, -- Current to a fan located on or serving the Left Engine.
    mcu_cpu_temp DOUBLE, -- Temperature of the CPU within the Main Control Unit.
    mcu_cpu_usage DOUBLE, -- Computational resource utilization of the CPU within the Main Control Unit.
    mcu_state DOUBLE, -- Operational state or status code of the Main Control Unit.
    mgu_temp_fl1 DOUBLE, -- Temperature of the first Motor Generator Unit at the front-left (FL) position.
    mgu_temp_fl2 DOUBLE, -- Temperature of the second Motor Generator Unit at the front-left (FL) position.
    mgu_temp_fr1 DOUBLE, -- Temperature of the Motor Generator Unit at the front-right (FR) position.
    msgcntr DOUBLE, -- A message counter, possibly for CAN messages or data frames sent/received.
    penta_switch1_temp DOUBLE, -- Temperature of the first penta switch, which could be related to a specific electronic or electrical system.
    penta_switch2_temp DOUBLE, -- Temperature of the second penta switch, referencing the same or related system as the first.
    reply_car_id DOUBLE, -- Identification of the car responding, possibly used in multi-car data acquisition.
    spring_travel_fl DOUBLE, -- Distance traveled by the suspension spring at the front-left wheel.
    spring_travel_fr DOUBLE, -- Distance traveled by the suspension spring at the front-right wheel.
    status DOUBLE, -- General status indicator, likely representing a system or component health check.
    steerangle_1 DOUBLE, -- Steering angle from the first sensor, indicating the position of the steering wheel or wheels.
    steerangle_2 DOUBLE, -- Steering angle from the second sensor, perhaps for redundancy or enhanced accuracy.
    synchronisation_id DOUBLE, -- An ID used to synchronize the dataset with other systems or data streams.
    throttle_percent_1 DOUBLE, -- Throttle application as a percentage from the primary sensor or control system.
    throttle_percent_2 DOUBLE, -- Throttle application as a percentage from the secondary sensor or control system.
    throttle_position_1 DOUBLE, -- Position of the throttle from the main sensor, reflecting the degree of opening.
    throttle_position_2 DOUBLE, -- Position of the throttle from a backup sensor or additional control input.
    v_bat_pdu DOUBLE, -- Voltage of the battery as measured by the Power Distribution Unit (PDU).
    voltage DOUBLE, -- General voltage measurement, additional details needed for specific context.
);

### Response:
Based on your instructions, here is the SQL query I have generated to answer the following question:
{question}

"""


def get_sql_chain(llm: LLM) -> Runnable:
    prompt: PromptTemplate = PromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt.partial(filename=str(settings.sql_dataset_path))  # type: ignore
    chain = RunnableLambda(lambda x: {"question": x}) | prompt | llm | StrOutputParser()
    return chain
