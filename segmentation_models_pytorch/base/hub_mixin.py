import json
from pathlib import Path
from typing import Optional, Union
from functools import wraps
from huggingface_hub import (
    PyTorchModelHubMixin,
    ModelCard,
    ModelCardData,
    hf_hub_download,
)


MODEL_CARD = """
---
{{ card_data }}
---
# {{ model_name }} Model Card

Table of Contents:
- [Load trained model](#load-trained-model)
- [Model init parameters](#model-init-parameters)
- [Model metrics](#model-metrics)
- [Dataset](#dataset)

## Load trained model
```python
import segmentation_models_pytorch as smp

model = smp.{{ model_name }}.from_pretrained("{{ save_directory | default("<save-directory-or-repo>", true)}}")
```

## Model init parameters
```python
model_init_params = {{ model_parameters }}
```

## Model metrics
{{ metrics | default("[More Information Needed]", true) }}

## Dataset
Dataset name: {{ dataset | default("[More Information Needed]", true) }}

## More Information
- Library: {{ repo_url | default("[More Information Needed]", true) }}
- Docs: {{ docs_url | default("[More Information Needed]", true) }}

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)
"""


def _format_parameters(parameters: dict):
    params = {k: v for k, v in parameters.items() if not k.startswith("_")}
    params = [
        f'"{k}": {v}' if not isinstance(v, str) else f'"{k}": "{v}"'
        for k, v in params.items()
    ]
    params = ",\n".join([f"    {param}" for param in params])
    params = "{\n" + f"{params}" + "\n}"
    return params


class SMPHubMixin(PyTorchModelHubMixin):
    def generate_model_card(self, *args, **kwargs) -> ModelCard:
        model_parameters_json = _format_parameters(self._hub_mixin_config)
        directory = self._save_directory if hasattr(self, "_save_directory") else None
        repo_id = self._repo_id if hasattr(self, "_repo_id") else None
        repo_or_directory = repo_id if repo_id is not None else directory

        metrics = self._metrics if hasattr(self, "_metrics") else None
        dataset = self._dataset if hasattr(self, "_dataset") else None

        if metrics is not None:
            metrics = json.dumps(metrics, indent=4)
            metrics = f"```json\n{metrics}\n```"

        model_card_data = ModelCardData(
            languages=["python"],
            library_name="segmentation-models-pytorch",
            license="mit",
            tags=["semantic-segmentation", "pytorch", "segmentation-models-pytorch"],
            pipeline_tag="image-segmentation",
        )
        model_card = ModelCard.from_template(
            card_data=model_card_data,
            template_str=MODEL_CARD,
            repo_url="https://github.com/qubvel/segmentation_models.pytorch",
            docs_url="https://smp.readthedocs.io/en/latest/",
            model_parameters=model_parameters_json,
            save_directory=repo_or_directory,
            model_name=self.__class__.__name__,
            metrics=metrics,
            dataset=dataset,
        )
        return model_card

    def _set_attrs_from_kwargs(self, attrs, kwargs):
        for attr in attrs:
            if attr in kwargs:
                setattr(self, f"_{attr}", kwargs.pop(attr))

    def _del_attrs(self, attrs):
        for attr in attrs:
            if hasattr(self, f"_{attr}"):
                delattr(self, f"_{attr}")

    @wraps(PyTorchModelHubMixin.save_pretrained)
    def save_pretrained(
        self, save_directory: Union[str, Path], *args, **kwargs
    ) -> Optional[str]:
        # set additional attributes to be used in generate_model_card
        self._save_directory = save_directory
        self._set_attrs_from_kwargs(["metrics", "dataset"], kwargs)

        # set additional attribute to be used in from_pretrained
        self._hub_mixin_config["_model_class"] = self.__class__.__name__

        try:
            # call the original save_pretrained
            result = super().save_pretrained(save_directory, *args, **kwargs)
        finally:
            # delete the additional attributes
            self._del_attrs(["save_directory", "metrics", "dataset"])
            self._hub_mixin_config.pop("_model_class")

        return result

    @wraps(PyTorchModelHubMixin.push_to_hub)
    def push_to_hub(self, repo_id: str, *args, **kwargs):
        self._repo_id = repo_id
        self._set_attrs_from_kwargs(["metrics", "dataset"], kwargs)
        result = super().push_to_hub(repo_id, *args, **kwargs)
        self._del_attrs(["repo_id", "metrics", "dataset"])
        return result

    @property
    def config(self):
        return self._hub_mixin_config


@wraps(PyTorchModelHubMixin.from_pretrained)
def from_pretrained(pretrained_model_name_or_path: str, *args, **kwargs):
    config_path = hf_hub_download(
        pretrained_model_name_or_path,
        filename="config.json",
        revision=kwargs.get("revision", None),
    )
    with open(config_path, "r") as f:
        config = json.load(f)
    model_class_name = config.pop("_model_class")

    import segmentation_models_pytorch as smp

    model_class = getattr(smp, model_class_name)
    return model_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
