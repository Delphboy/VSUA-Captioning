from __future__ import absolute_import, division, print_function

from .VSUAModel import VSUAModel


# TODO: Convert this into an actual factory pattern
def setup(opt):
    if opt.caption_model == "vsua":
        model = VSUAModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model
