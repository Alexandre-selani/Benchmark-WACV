"""
    Import aliases that let the released GFROR checkpoints unpickle.

    The GFROR checkpoints were produced with `torch.save(module)` rather than
    `torch.save(module.state_dict())`, so each file stores the *import path* of
    the classes it contains — `model.vanilla_ae.VanillaAE64`, and so on. In this
    repository that package lives at `methods.gfror.model`, so unpickling would
    fail with `ModuleNotFoundError: No module named 'model'`.

    Registering the old names in sys.modules makes the pickles resolve without
    touching the checkpoint files, which keeps them bit-identical to the ones
    the reported results were produced with.
"""

import sys

from . import model as _gfror_model

_ALIASES = ["classifier", "decoder", "encoder", "utils", "vanilla_ae"]


def install():
    sys.modules.setdefault("model", _gfror_model)
    for name in _ALIASES:
        sys.modules.setdefault(f"model.{name}", getattr(_gfror_model, name, None)
                               or __import__(f"methods.gfror.model.{name}", fromlist=[name]))
