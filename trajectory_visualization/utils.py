from ai2thor.controller import Controller
import os
import numpy as np
import cv2
import json


class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position
        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )


def controller_init_(scene_name='FloorPlan_Val2_3'):
    c = Controller(scene=scene_name,width=800,height=800)
    return c


def make_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        print('make new dirs ', dirs)
    else:
        print(dirs, ' has existed')


def clear_files(dirs):
    for file in os.listdir(dirs):
        os.remove(os.path.join(dirs, file))
    print('the file in {} has been removed'.format(dirs))


def action(c,action = None):
    actions = ['MoveAhead','RotateLeft','RotateRight','LookUp','LookDown','Done']
    c.step(dict(action=actions[action],degrees=45,gridSize=0.25))


def position_to_tuple(position):
    return (position["x"], position["y"], position["z"])


def agent_point(c):
    c.step({"action": "ToggleMapView"})
    position = c.last_event.metadata['agent']['position']
    cam_position = c.last_event.metadata["cameraPosition"]
    cam_orth_size = c.last_event.metadata["cameraOrthSize"]
    pos_translator = ThorPositionTo2DFrameTranslator(
        c.last_event.frame.shape, position_to_tuple(cam_position), cam_orth_size
    )
    point = pos_translator(np.array((position['x'], position['z'])))
    c.step({"action": "ToggleMapView"})
    return tuple(reversed(point))


def get_draw_picture(c,start_state,actions):
    c.step(dict(action='TeleportFull',
                x=start_state['x'],
                y=start_state['y'],
                z=start_state['z'],
                rotation=start_state['rotation'],
                horizon=start_state['horizon']))
    for a in actions:
        action(c, action=a)
    c.step({"action": "ToggleMapView"})
    topview = c.last_event.cv2img
    c.step({"action": "ToggleMapView"})
    return topview

def strcolor2tuplecolor(color):
    b = int(color[:2],16)
    g = int(color[2:4], 16)
    r = int(color[4:6], 16)
    return (r,g,b)


