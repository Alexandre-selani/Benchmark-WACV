"""
    Import aliases that let the released GFROR checkpoints unpickle.

    The GFROR checkpoints were produced with `torch.save(module)` rather than
    `torch.save(module.state_dict())`, so each file stores the *import path* of
    the classes it contains. Two of those paths no longer exist:

        model.vanilla_ae.VanillaAE64     the autoencoder, now methods.gfror.model
        Modelos.ResNet18_backbone.*      the classifier, now the Models package

    Registering the old names in sys.modules makes the pickles resolve without
    touching the checkpoint files, which keeps them bit-identical to the ones
    the reported results were produced with.

    Without the `Modelos` alias the failure is loud on a clean environment, but
    silent on one that still has an older package of that name installed — the
    class would be taken from there instead. The alias removes that ambiguity.
"""

import sys

import Models

from . import model as _gfror_model

_MODEL_SUBMODULES = ["classifier", "decoder", "encoder", "utils", "vanilla_ae"]


def install():
    # The autoencoder's pre-refactor package.
    sys.modules.setdefault("model", _gfror_model)
    for name in _MODEL_SUBMODULES:
        sys.modules.setdefault(
            f"model.{name}",
            getattr(_gfror_model, name, None)
            or __import__(f"methods.gfror.model.{name}", fromlist=[name]))

    # The classifier's pre-refactor backbone package, renamed Modelos -> Models.
    sys.modules.setdefault("Modelos", Models)
    for name in ("ResNet18_backbone", "Base_CAC"):
        sys.modules.setdefault(f"Modelos.{name}",
                               __import__(f"Models.{name}", fromlist=[name]))
