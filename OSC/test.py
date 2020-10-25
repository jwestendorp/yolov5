from pythonosc.udp_client import SimpleUDPClient

ip = "127.0.0.1"
port = 1337

client = SimpleUDPClient(ip, port)  # Create client

# client.send_message("/some/address", 123)   # Send float message
# Send message with int, float and string
# client.send_message("/some/address", [1, 2., "hello"])

# x,y,w,h
client.send_message("/some/address", [10, 20, 250, 250])
