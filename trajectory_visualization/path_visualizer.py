import os.path

from utils import *
import cv2
import copy
import operator
import numpy as np
class PathVisualizer(object):
    def __init__(self,json_path):
        self.c = controller_init_()
        self.eps = self.read_json(json_path)
        self.start_state = None
        self.actions = None
        self.scene = None
        self.target = None
        self.back_ground_pic = None
        self.path_pic = None

    @staticmethod
    def read_json(path):
        eps = []
        if '.json' in path:
            with open(path, 'r') as rf:
                data = json.load(rf)
            eps += data['episodes']
        else:
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r') as rf:
                    data = json.load(rf)
                eps += data['episodes']
        return eps

    def filter(self):
        eps = []
        for ep in self.eps:
            #filter rule
            ############
            eps.append(ep)
        self.eps = eps

    def draw_path(self):
        self.c.step(dict(action='TeleportFull',
                    x=self.start_state['x'],
                    y=self.start_state['y'],
                    z=self.start_state['z'],
                    rotation=self.start_state['rotation'],
                    horizon=self.start_state['horizon']))
        self.path_pic=copy.deepcopy(self.back_ground_pic)
        point = agent_point(self.c)
        for i, a in enumerate(self.actions):
            action(self.c, action=a)
            new_point = agent_point(self.c)
            if not operator.eq(point,new_point):
                d = np.linalg.norm(np.array(point)-np.array(new_point))
                arrow_l = 15
                dx = int(arrow_l*(new_point[0]-point[0])/d)
                dy = int(arrow_l*(new_point[1]-point[1])/d)
                end = (point[0]+dx,point[1]+dy)
                self.path_pic = cv2.arrowedLine(self.path_pic.astype(np.uint8), point,end,color=(0,255,0), thickness=5,tipLength=0.4)
            point = new_point
    def reset(self):
        self.c.reset(self.scene)
        self.c.step(dict(action='Initialize',
                    gridSize=0.25,
                    degrees=45,
                    continuous=True,
                    renderObjectImage=True))
    def draw_ep(self,ep):
        self.start_state = ep['start_state']
        self.actions = ep['actions']
        self.scene = ep['scene']
        self.target = ep['target']
        self.reset()
        # self.c.reset(self.scene)
        # self.c.step(dict(action='Initialize',
        #             gridSize=0.25,
        #             degrees=45,
        #             continuous=True,
        #             renderObjectImage=True))
        self.back_ground_pic = get_draw_picture(self.c, self.start_state, self.actions)
        self.draw_path()

    def save(self,save_dir):
        if not os.path.exists(save_dir):
            make_dirs(save_dir)
        file = self.scene+self.target+'_'+str(self.start_state['x'])+'_' +\
            str(self.start_state['z'])+'_'+str(self.start_state['rotation']['y'])+'.jpg'
        save_path = os.path.join(save_dir, file)
        cv2.imwrite(save_path, self.path_pic)

    def visualize_path(self, save_dir):
        self.filter()
        num = 0
        for ep in self.eps:
            num += 1
            self.draw_ep(ep)
            self.save(save_dir)
            print('finish', num, 'all', len(self.eps))


