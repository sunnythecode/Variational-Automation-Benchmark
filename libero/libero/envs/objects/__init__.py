from libero.envs.base_object import OBJECTS_DICT, VISUAL_CHANGE_OBJECTS_DICT

from .hope_objects import *
from .google_scanned_objects import *
from .articulated_objects import *
from .turbosquid_objects import *
from .self_designed_object import RedSticker, LiberoMugYellow


def get_object_fn(category_name):
    return OBJECTS_DICT[category_name.lower()]


def get_object_dict():
    return OBJECTS_DICT
