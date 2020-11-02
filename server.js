const { Client, Server } = require("./node_modules/node-osc/dist/lib");
const express = require("express");
const http = require("http");
const socketIO = require("socket.io");
const app = express();
let server = http.createServer(app);
let io = socketIO(server);

// const client = new Client("127.0.0.1", 1337);
var osc = new Server(1337, "0.0.0.0");

// server.on("listening", () => {
//   console.log("OSC Server is listening.");
// });

osc.on("bird", (msg) => {
  console.log(`Message: ${msg}`);
});

// client.send("/hello", "world", (err) => {
//   if (err) console.error(err);
//   client.close();
// });

function newSocketConnection(socket) {
  let { id } = socket;

  console.log(id, "connected");
}

io.on("connection", newSocketConnection);

server.listen(8080, () => {
  console.log("Your app is listening on port ");
});
