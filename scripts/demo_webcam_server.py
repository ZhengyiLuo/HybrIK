#!/usr/bin/env python3
import os
import cv2
import joblib
import numpy as np
import time

import asyncio
from aiohttp import web
import cv2
import aiohttp
import numpy as np
import threading
from scipy.spatial.transform import Rotation as sRot

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d
import time
from easydict import EasyDict as edict
import torch


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


async def main():
    global pose_mat, trans, dt, reset_offset
    offset = 0
    cfg_file = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix_w_pw3d_fast.yaml'
    CKPT = './pretrained_w_cam.pth'
    cfg = update_config(cfg_file)

    bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
    bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
    dummpy_set = edict({'joint_pairs_17': None, 'joint_pairs_24': None, 'joint_pairs_29': None, 'bbox_3d_shape': bbox_3d_shape})

    transformation = SimpleTransform3DSMPLCam(dummpy_set,
                                              scale_factor=cfg.DATASET.SCALE_FACTOR,
                                              color_factor=cfg.DATASET.COLOR_FACTOR,
                                              occlusion=cfg.DATASET.OCCLUSION,
                                              input_size=cfg.MODEL.IMAGE_SIZE,
                                              output_size=cfg.MODEL.HEATMAP_SIZE,
                                              depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                                              bbox_3d_shape=bbox_3d_shape,
                                              rot=cfg.DATASET.ROT_FACTOR,
                                              sigma=cfg.MODEL.EXTRA.SIGMA,
                                              train=False,
                                              add_dpg=False,
                                              loss_type=cfg.LOSS['TYPE'])

    det_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    det_model.classes = [0]

    hybrik_model = builder.build_sppe(cfg.MODEL)

    print(f'Loading model from {CKPT}...')
    save_dict = torch.load(CKPT, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict)
    det_model.cuda()
    hybrik_model.cuda()
    det_model.eval()
    hybrik_model.eval()
    device = torch.device('cuda')

    from scipy.spatial.transform import Rotation as sRot
    global_transform = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv().as_matrix()
    transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()

    def frames_from_webcam():
        cap = cv2.VideoCapture(-1)

        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            yield frame[..., ::-1]

    prev_box = None

    print('### Run Model...')
    for frame in frames_from_webcam():
        with torch.no_grad():
            # Run Detection
            t_s = time.time()

            yolo_output = det_model([frame], size=280)

            yolo_out_xyxy = torch.stack(yolo_output.xyxy)
            if yolo_out_xyxy.shape[1] > 0:
                det_output = {"boxes": yolo_out_xyxy[:, 0, :4], "scores": yolo_out_xyxy[:, 0, 4]}

                if prev_box is None:
                    tight_bbox = get_one_box(det_output)  # xyxy
                    if tight_bbox is None:
                        continue
                else:
                    tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy

                prev_box = tight_bbox

                # Run HybrIK
                # bbox: [x1, y1, x2, y2]
                pose_input, bbox, img_center = transformation.test_transform(frame, tight_bbox)
                pose_input = pose_input.to(device)[None, :, :, :]

                pose_output = hybrik_model(pose_input, flip_test=False, bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(), img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float())

                ######## Streaming models
                pose_mat = pose_output.pred_theta_mats.cpu()
                trans = pose_output.transl.cpu().numpy()
                scale = (bbox[2] - bbox[0]) / 256

                trans[:, 2] /= scale
                if reset_offset:
                    offset = -0.89 - trans[0, 1]
                    reset_offset = False
                trans[:, 1] += offset

                # print(trans, trans.dot(sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix().T))
                ######## Streaming models

                dt = time.time() - t_s
                print(f'\r {1/dt:.2f} fps', end='')

            else:
                print("no human detected~")


def frames_from_webcam():
    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        yield frame[..., ::-1]


async def pose_getter(request):
    # query env configurations
    global pose_mat, trans, dt
    curr_paths = {}

    json_resp = {
        "pose_mat": pose_mat.tolist(),
        "trans": trans.tolist(),
        "dt": dt,
    }

    return web.json_response(json_resp)


async def websocket_handler(request):
    print('Websocket connection starting')
    global pose_mat, trans, dt
    ws_talker = aiohttp.web.WebSocketResponse()

    await ws_talker.prepare(request)
    print('Websocket connection ready')

    async for msg in ws_talker:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == "get_pose":
                await ws_talker.send_json({
                    "pose_mat": pose_mat.tolist(),
                    "trans": trans.tolist(),
                    "dt": dt,
                })

    print('Websocket connection closed')
    return ws_talker


async def talk_websocket_handler(request):
    print('Websocket connection starting')
    global reset_offset, trans
    ws_talker = aiohttp.web.WebSocketResponse()

    await ws_talker.prepare(request)
    print('Websocket connection ready')

    async for msg in ws_talker:
        #		print(msg)
        if msg.type == aiohttp.WSMsgType.TEXT:
            print(msg.data)
            if msg.data == "reset_pose":
                reset_offset = True
                await ws_talker.send_str("fine!")

    print('Websocket connection closed')
    return ws_talker


def start_pose_estimate():
    loop = asyncio.new_event_loop()  # <-- create new loop in this thread here
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())


pose_mat, trans, dt, ws_talkers, reset_offset = np.zeros([24, 3, 3]), np.zeros([3]), 1 / 10, [], 0
# main()
app = web.Application(client_max_size=1024**2)
app.router.add_route('GET', '/ws', websocket_handler)
app.router.add_route('GET', '/ws_talk', talk_websocket_handler)
app.router.add_route('GET', '/get_pose', pose_getter)
threading.Thread(target=start_pose_estimate, daemon=True).start()

web.run_app(app, port=8080)
