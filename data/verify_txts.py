"""verify_txts.py

For verifying correctness of the generated YOLO txt annotations.
"""


import random
from pathlib import Path
import os
from argparse import ArgumentParser
from distutils.util import strtobool

import cv2


WINDOW_NAME = "verify_txts"

parser = ArgumentParser()
parser.add_argument('--dim', type=str, default="416x416", help='input width and height, e.g. 608x608')
parser.add_argument('--server', type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument('--num', type=int, default=5)
args = parser.parse_args()

print('Verifying train.txt')
jpgs_path = Path('crowdhuman-%s/train.txt' % args.dim)

if args.server:
    render_jpgs_path = 'crowdhuman-%s/verify' % args.dim
    os.makedirs(render_jpgs_path, exist_ok=True)

with open(jpgs_path.as_posix(), 'r') as f:
    jpg_names = [l.strip()[5:] for l in f.readlines()][:args.num] # remove "data/""

random.shuffle(jpg_names)
for jpg_name in jpg_names:
    img = cv2.imread(jpg_name)
    img_h, img_w, _ = img.shape
    txt_name = jpg_name.replace('.jpg', '.txt').replace('images', 'labels')
    with open(txt_name, 'r') as f:
        obj_lines = [l.strip() for l in f.readlines()]
    for obj_line in obj_lines:
        cls, cx, cy, nw, nh = [float(item) for item in obj_line.split(' ')]
        color = (0, 0, 255) if cls == 0.0 else (0, 255, 0)
        x_min = int((cx - (nw / 2.0)) * img_w)
        y_min = int((cy - (nh / 2.0)) * img_h)
        x_max = int((cx + (nw / 2.0)) * img_w)
        y_max = int((cy + (nh / 2.0)) * img_h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    if args.server: 
        cv2.imwrite(os.path.join(render_jpgs_path, jpg_name.split('/')[-1]), img)
    else:
        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(0) == 27:
            break

cv2.destroyAllWindows()
