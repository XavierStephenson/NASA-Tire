import cv2
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
from menu_functions import *

import settings
import Extract
import View
import LiveView
import FindColor
settings.init()
if __name__ == '__main__':
    mp.freeze_support()
    while True:
        i = ClickMenu(settings.window_name, 'Main Menu', ["Extract Data", "View Stored Data", "Live Viewer", "Find Color", "Options"], True)
        match i:
            case False: break
            case 1: Extract.init()
            case 2: View.init()
            case 3: LiveView.init()
            case 4: FindColor.init()