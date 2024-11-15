import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import hydra 
import json 

@hydra.main(version_base=None, config_path='./config', config_name='base')
def main(args):
    # Mosquitto MQTT Configuration
    MQTT_BROKER = args.MQTT.BROKER_ADDRESS  # Replace with your MQTT broker address
    MQTT_PORT = 1883
    MQTT_TOPIC = "AUDIOLOGGER"  # Example topic to subscribe to

    # InfluxDB Configuration
    INFLUXDB_URL = args.INFLUXDB.URL 
    INFLUXDB_TOKEN = args.INFLUXDB.TOKEN
    INFLUXDB_ORG = args.INFLUXDB.ORG
    INFLUXDB_BUCKET = args.INFLUXDB.BUCKET

    # Initialize InfluxDB Client
    influxdb_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN)
    write_api = influxdb_client.write_api(write_options=SYNCHRONOUS)

    # Callback when the client connects to the broker
    def on_connect(client, userdata, flags, rc):
        print(f"Connected to MQTT Broker with code {rc}")
        client.subscribe(MQTT_TOPIC)

    # Callback when a message is received from the broker
    def on_message(client, userdata, msg):
        try:
            # Decode and parse the message
            payload = msg.payload.decode()
            # print(f"Received message: {payload} on topic {msg.topic}")
            
            # Convert the payload to a data point
            # Example assumes JSON payload with fields 'temperature' and 'timestamp'
            data = json.loads(payload)  # Replace eval with a safer parsing method (e.g., json.loads) for JSON
            
            # Create a Point for InfluxDB
            point = (
                Point("audiolog")  # Measurement name
                .field("audio", str(data["data"]))  # Field
                .time(data["timestamp"])  # Timestamp
            )
            
            # Write to InfluxDB
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
            print(f"Written to InfluxDB: {point}")
        except Exception as e:
            print(f"Error processing message: {e}")

    # Initialize MQTT Client
    mqtt_client = mqtt.Client()

    # Assign callbacks
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    # Connect to MQTT Broker
    print("Connecting to MQTT Broker...")
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

    # Start MQTT loop
    mqtt_client.loop_forever()

if __name__=="__main__":
    main()