from pythonosc.udp_client import SimpleUDPClient

ip = "127.0.0.1"
port = 1337

client = SimpleUDPClient(ip, port)  # Create client

# client.send_message("/some/address", 123)   # Send float message
# Send message with int, float and string
# client.send_message("/some/address", [1, 2., "hello"])


client.send_message("/some/address", [0.91, 0.99, 1, 12])
