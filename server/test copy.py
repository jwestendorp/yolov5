from aiohttp import web
import socketio

sio = socketio.AsyncServer()
# app = web.Application()
# sio.attach(app)

static_files = {
    '/static/socket.io.js': 'static/socket.io.js',
}

app = socketio.ASGIApp(sio, static_files=static_files)


# async def index(request):
#     return web.Response(text='hello', content_type='text/html')


@sio.event
def connect(sid, environ):
    print("connect ", sid)


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


# app.router.add_static('/static', 'static')
# app.router.add_get('/', index)

if __name__ == '__main__':
    web.run_app(app)
