import argparse
import os
import shutil
import time
import eventlet
import socketio
import asyncio
from aiohttp import web
from pathlib import Path
from multiprocessing import Process
import threading
import nest_asyncio
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

nest_asyncio.apply()


# sio = socketio.AsyncServer()
# app = web.Application()
# sio.attach(app)


# async def index(request):
#     """Serve the client-side application."""

#     with open('server/index.html') as f:
#         return web.Response(text=f.read(), content_type='text/html')


# @sio.event
# def connect(sid, environ):
#     print("connect ", sid)


# @sio.event
# def disconnect(sid):
#     print('disconnect ', sid)


# app.router.add_static('/static', 'server')
# app.router.add_get('/', index)


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(
        ('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    ########
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if det is not None:
                birds = torch.tensor(list(
                    filter(lambda x: names[int(x[-1])] == "bird", det.cpu().numpy())))

                if birds is not None and len(birds):

                    # print(len(birds), 'birds', birds)
                    # print(len(det), 'det', det)

                    # Rescale boxes from img_size to im0 size
                    birds[:, :4] = scale_coords(
                        img.shape[2:], birds[:, :4], im0.shape).round()

                    # # Write results
                    for *xyxy, conf, cls in reversed(birds):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh

                        print(xywh)
                        # client.send_message("/bird", xywh)

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label,
                                         color=colors[int(cls)], line_thickness=3)

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov5s.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='output confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    opt = parser.parse_args()

    # async def start_server():
    #     web.run_app(app)

    def start_model():
        with torch.no_grad():
            # update all models (to fix SourceChangeWarning)
            if opt.update:
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    detect()
                    strip_optimizer(opt.weights)

            else:
                detect()

    start_model()

    # async def main():

    #     loop = asyncio.get_running_loop()

    #     task = loop.create_task(start_server())
    #     # task2 = loop.create_task(start_server)

    #     # thread = threading.Thread(target=start_server)
    #     # thread2 = threading.Thread(target=start_model)
    #     # thread.start()
    #     # thread2.start()
    #     # Schedule three calls *concurrently*:
    #     # await asyncio.gather(
    #     #     start_model(),
    #     #     start_server(),
    #     # )

    # asyncio.run(main())
