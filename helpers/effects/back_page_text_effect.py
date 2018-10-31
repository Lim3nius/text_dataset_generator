
import sys
import copy
import random
import numpy as np
import traceback

import helper
from helpers import image_helper
from helpers import text_renderer


def generate_back_image(font, config):
    complete_content = copy.deepcopy(config["OriginalText"][0])
    selected_content = complete_content[np.random.randint(len(complete_content)):]

    back_text_img = None

    try:
        back_text_img, _, _, _ = text_renderer.render_page(font, [selected_content], config)
        back_text_img = np.flip(back_text_img, axis=1)
    except:
        traceback.print_exc()
    
    return back_text_img





