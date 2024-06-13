
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from models.common import DetectMultiBackend

class YoloROS(object):
    def __init__(self, weight_path) -> None:
        self.device = torch.device("cuda")
        self.model = DetectMultiBackend(weight_path, device=self.device, dnn=False, data=ROOT.joinpath("data", "coco.yaml"), fp16=False)
        pass

if __name__ == "__main__":
    weight_path = ROOT.joinpath("weights", "yolov9-m-converted.pt")
    yr = YoloROS(weight_path)
    pass