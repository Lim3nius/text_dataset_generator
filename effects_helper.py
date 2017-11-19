
from Effects import surrounding_text_effect

def apply_effects(img, text, words_dict, config):
    img = surrounding_text_effect.apply_effect(img, text, words_dict, config)
    return img
