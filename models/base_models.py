import re
import sys
import functools
sys.path.append('CLIP/')
import torch
from CLIP import clip
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.check_submission import check_models
"""
Template module for a base model submission to brain-score
"""

def get_model_list():
    #return ['ViT-B/32', 'RN50']
    return ['ViT-B/32']


def get_model(name):
    assert name in ['ViT-B/32', 'RN50']
    model, _ = clip.load(name, jit=False)
    model = model.visual
    model.to(torch.float32, non_blocking=False)    # cast all weights from HalfTensors to FloatTensor
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='clip', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    if name == 'ViT-B/32':
        num_layers = 12
        layers = [f'transformer.resblocks.{i}.ln_2' for i in range(num_layers)]
    return layers


def get_bibtex(model_identifier):
    return """@article{radford2learning,
                    title={Learning Transferable Visual Models From Natural Language Supervision},
                    author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
                    journal={Image},
                    volume={2},
                    pages={T2}}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
