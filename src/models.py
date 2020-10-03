"""
models.py

PURPOSE: This file defines Neural Network Architecture and other models
        which will be evaluated in this expirement
"""
import torch
from os import path


def save_model(model):
    # if isinstance(model, Planner):
    #     return save(
    #       model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th')
    #     )
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    r = None
    # if isinstance(model, Planner):
    #     r = Planner()
    #     r.load_state_dict(load(
    #         path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location=model.device)
    #     )
    return r
