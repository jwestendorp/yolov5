import paho.mqtt.client as mqtt
import json
import time

client = mqtt.Client("jonaslappy")
client.username_pw_set("80af5a0a", "548b12d6693ed913")
client.connect("broker.shiftr.io", 1883)

while True:
    payload = json.dumps(
        {'x': 23, 'y': 34, 'w': 300, 'h': 300, 'confidence': .8})
    res = client.publish("birdBoundingBox/birdBoundingBox", payload)
    print("published")
    time.sleep(1)
