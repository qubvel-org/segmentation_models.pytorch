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

model = smp.from_pretrained("<save-directory-or-this-repo>")
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
        model_parameters_json = _format_parameters(self.config)
        metrics = kwargs.get("metrics", None)
        dataset = kwargs.get("dataset", None)

        if metrics is not None:
            metrics = json.dumps(metrics, indent=4)
            metrics = f"```json\n{metrics}\n```"

        tags = self._hub_mixin_info.model_card_data.get("tags", []) or []
        tags.extend(["segmentation-models-pytorch", "semantic-segmentation", "pytorch"])

        model_card_data = ModelCardData(
            languages=["python"],
            library_name="segmentation-models-pytorch",
            license="mit",
            tags=tags,
            pipeline_tag="image-segmentation",
        )
        model_card = ModelCard.from_template(
            card_data=model_card_data,
            template_str=MODEL_CARD,
            repo_url="https://github.com/qubvel/segmentation_models.pytorch",
            docs_url="https://smp.readthedocs.io/en/latest/",
            model_parameters=model_parameters_json,
            model_name=self.__class__.__name__,
            metrics=metrics,
            dataset=dataset,
        )
        return model_card

    @wraps(PyTorchModelHubMixin.save_pretrained)
    def save_pretrained(
        self, save_directory: Union[str, Path], *args, **kwargs
    ) -> Optional[str]:
        model_card_kwargs = kwargs.pop("model_card_kwargs", {})
        if "dataset" in kwargs:
            model_card_kwargs["dataset"] = kwargs.pop("dataset")
        if "metrics" in kwargs:
            model_card_kwargs["metrics"] = kwargs.pop("metrics")
        kwargs["model_card_kwargs"] = model_card_kwargs

        # set additional attribute to be able to deserialize the model
        self.config["_model_class"] = self.__class__.__name__

        try:
            # call the original save_pretrained
            result = super().save_pretrained(save_directory, *args, **kwargs)
        finally:
            self.config.pop("_model_class", None)

        return result

    @property
    def config(self) -> dict:
        return self._hub_mixin_config


@wraps(PyTorchModelHubMixin.from_pretrained)
def from_pretrained(pretrained_model_name_or_path: str, *args, **kwargs):
    config_path = Path(pretrained_model_name_or_path) / "config.json"
    if not config_path.exists():
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
