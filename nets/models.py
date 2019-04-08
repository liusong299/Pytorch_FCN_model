import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

from .fcn import *

logging.info("Get pretrained VGG ......")
vgg_model = VGGNet(requires_grad=True)

all_models = {
    'fcn8': lambda n_class: FCN8s(pretrained_net=vgg_model, n_class=n_class),
    'fcn16': lambda n_class: FCN16s(pretrained_net=vgg_model, n_class=n_class),
    'fcn32': lambda n_class: FCN32s(pretrained_net=vgg_model, n_class=n_class),
}
