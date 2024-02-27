import json

import timm
from tqdm import tqdm

if __name__ == "__main__":
    # Check for models that support `features_only=True``
    works, fails = {}, []
    for model in tqdm(timm.list_models()):
        try:
            m = timm.create_model(model, pretrained=False, features_only=True)
            works[model] = dict(
                indices=m.feature_info.out_indices,
                channels=m.feature_info.channels(),
                reduction=m.feature_info.reduction(),
                module=m.feature_info.module_name(),
            )
        except RuntimeError:
            fails.append(model)

    with open("encoders_features_only_supported.json", "w") as f:
        json.dump(works, f, indent=2)

    # Check for models that support `get_intermediate_layers``
    intermediate_layers_support = []
    unsupported = []

    for model in tqdm(fails):
        m = timm.create_model(model, pretrained=False)
        if hasattr(m, "get_intermediate_layers"):
            intermediate_layers_support.append(model)
        else:
            unsupported.append(model)

    with open("encoders_get_intermediate_layers_supported.json", "w") as f:
        json.dump(intermediate_layers_support, f, indent=2)

    # Save unsupported timm models
    with open("encoders_unsupported.json", "w") as f:
        json.dump(unsupported, f, indent=2)
