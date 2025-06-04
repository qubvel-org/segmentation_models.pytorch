import re
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download, HfApi
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fmt: off
CONVNEXT_MAPPING = {
    r"backbone.embeddings.patch_embeddings.(weight|bias)":                   r"encoder.model.stem_0.\1",
    r"backbone.embeddings.layernorm.(weight|bias)":                          r"encoder.model.stem_1.\1",
    r"backbone.encoder.stages.(\d+).layers.(\d+).layer_scale_parameter":     r"encoder.model.stages_\1.blocks.\2.gamma",
    r"backbone.encoder.stages.(\d+).layers.(\d+).dwconv.(weight|bias)":      r"encoder.model.stages_\1.blocks.\2.conv_dw.\3",
    r"backbone.encoder.stages.(\d+).layers.(\d+).layernorm.(weight|bias)":   r"encoder.model.stages_\1.blocks.\2.norm.\3",
    r"backbone.encoder.stages.(\d+).layers.(\d+).pwconv(\d+).(weight|bias)": r"encoder.model.stages_\1.blocks.\2.mlp.fc\3.\4",
    r"backbone.encoder.stages.(\d+).downsampling_layer.(\d+).(weight|bias)": r"encoder.model.stages_\1.downsample.\2.\3",
}

SWIN_MAPPING = {
    r"backbone.embeddings.patch_embeddings.projection":                                             r"encoder.model.patch_embed.proj",
    r"backbone.embeddings.norm":                                                                    r"encoder.model.patch_embed.norm",
    r"backbone.encoder.layers.(\d+).blocks.(\d+).layernorm_before":                                 r"encoder.model.layers_\1.blocks.\2.norm1",
    r"backbone.encoder.layers.(\d+).blocks.(\d+).attention.self.relative_position_bias_table":      r"encoder.model.layers_\1.blocks.\2.attn.relative_position_bias_table",
    r"backbone.encoder.layers.(\d+).blocks.(\d+).attention.self.(query|key|value)":                 r"encoder.model.layers_\1.blocks.\2.attn.\3",
    r"backbone.encoder.layers.(\d+).blocks.(\d+).attention.output.dense":                           r"encoder.model.layers_\1.blocks.\2.attn.proj",
    r"backbone.encoder.layers.(\d+).blocks.(\d+).layernorm_after":                                  r"encoder.model.layers_\1.blocks.\2.norm2",
    r"backbone.encoder.layers.(\d+).blocks.(\d+).intermediate.dense":                               r"encoder.model.layers_\1.blocks.\2.mlp.fc1",
    r"backbone.encoder.layers.(\d+).blocks.(\d+).output.dense":                                     r"encoder.model.layers_\1.blocks.\2.mlp.fc2",
    r"backbone.encoder.layers.(\d+).downsample.reduction":                                          lambda x: f"encoder.model.layers_{1 + int(x.group(1))}.downsample.reduction",
    r"backbone.encoder.layers.(\d+).downsample.norm":                                               lambda x: f"encoder.model.layers_{1 + int(x.group(1))}.downsample.norm",
}

DECODER_MAPPING = {

    # started from 1 in hf
    r"backbone.hidden_states_norms.stage(\d+)":                              lambda x: f"decoder.feature_norms.{int(x.group(1)) - 1}",

    r"decode_head.psp_modules.(\d+).(\d+).conv.weight":                      r"decoder.psp.blocks.\1.\2.0.weight",
    r"decode_head.psp_modules.(\d+).(\d+).batch_norm":                       r"decoder.psp.blocks.\1.\2.1",
    r"decode_head.bottleneck.conv.weight":                                   r"decoder.psp.out_conv.0.weight",
    r"decode_head.bottleneck.batch_norm":                                    r"decoder.psp.out_conv.1",

    # fpn blocks are in reverse order (3 blocks total, so 2 - i)
    r"decode_head.lateral_convs.(\d+).conv.weight":                          lambda x: f"decoder.fpn_lateral_blocks.{2 - int(x.group(1))}.conv_norm_relu.0.weight",
    r"decode_head.lateral_convs.(\d+).batch_norm":                           lambda x: f"decoder.fpn_lateral_blocks.{2 - int(x.group(1))}.conv_norm_relu.1",
    r"decode_head.fpn_convs.(\d+).conv.weight":                              lambda x: f"decoder.fpn_conv_blocks.{2 - int(x.group(1))}.0.weight",
    r"decode_head.fpn_convs.(\d+).batch_norm":                               lambda x: f"decoder.fpn_conv_blocks.{2 - int(x.group(1))}.1",

    r"decode_head.fpn_bottleneck.conv.weight":                              r"decoder.fusion_block.0.weight",
    r"decode_head.fpn_bottleneck.batch_norm":                               r"decoder.fusion_block.1",
    r"decode_head.classifier":                                              r"segmentation_head.0",
}
# fmt: on

PRETRAINED_CHECKPOINTS = {
    "convnext-tiny": {
        "repo_id": "openmmlab/upernet-convnext-tiny",
        "encoder_name": "tu-convnext_tiny",
        "decoder_channels": 512,
        "classes": 150,
        "mapping": {**CONVNEXT_MAPPING, **DECODER_MAPPING},
    },
    "convnext-small": {
        "repo_id": "openmmlab/upernet-convnext-small",
        "encoder_name": "tu-convnext_small",
        "decoder_channels": 512,
        "classes": 150,
        "mapping": {**CONVNEXT_MAPPING, **DECODER_MAPPING},
    },
    "convnext-base": {
        "repo_id": "openmmlab/upernet-convnext-base",
        "encoder_name": "tu-convnext_base",
        "decoder_channels": 512,
        "classes": 150,
        "mapping": {**CONVNEXT_MAPPING, **DECODER_MAPPING},
    },
    "convnext-large": {
        "repo_id": "openmmlab/upernet-convnext-large",
        "encoder_name": "tu-convnext_large",
        "decoder_channels": 512,
        "classes": 150,
        "mapping": {**CONVNEXT_MAPPING, **DECODER_MAPPING},
    },
    "convnext-xlarge": {
        "repo_id": "openmmlab/upernet-convnext-xlarge",
        "encoder_name": "tu-convnext_xlarge",
        "decoder_channels": 512,
        "classes": 150,
        "mapping": {**CONVNEXT_MAPPING, **DECODER_MAPPING},
    },
    "swin-tiny": {
        "repo_id": "openmmlab/upernet-swin-tiny",
        "encoder_name": "tu-swin_tiny_patch4_window7_224",
        "decoder_channels": 512,
        "classes": 150,
        "extra_kwargs": {"img_size": 512},
        "mapping": {**SWIN_MAPPING, **DECODER_MAPPING},
    },
    "swin-small": {
        "repo_id": "openmmlab/upernet-swin-small",
        "encoder_name": "tu-swin_small_patch4_window7_224",
        "decoder_channels": 512,
        "classes": 150,
        "extra_kwargs": {"img_size": 512},
        "mapping": {**SWIN_MAPPING, **DECODER_MAPPING},
    },
    "swin-large": {
        "repo_id": "openmmlab/upernet-swin-large",
        "encoder_name": "tu-swin_large_patch4_window12_384",
        "decoder_channels": 512,
        "classes": 150,
        "extra_kwargs": {"img_size": 512},
        "mapping": {**SWIN_MAPPING, **DECODER_MAPPING},
    },
}


def convert_old_keys_to_new_keys(state_dict_keys: dict, keys_mapping: dict):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in keys_mapping.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def group_qkv_layers(state_dict: dict) -> dict:
    """Find corresponding layer names for query, key and value layers and stack them in a single layer"""

    state_dict = state_dict.copy()  # shallow copy

    result = defaultdict(dict)
    layer_names = list(state_dict.keys())
    qkv_names = ["query", "key", "value"]
    for layer_name in layer_names:
        for pattern in qkv_names:
            if pattern in layer_name:
                new_key = layer_name.replace(pattern, "qkv")
                result[new_key][pattern] = state_dict.pop(layer_name)
                break

    # merge them all
    for new_key, patterns in result.items():
        state_dict[new_key] = torch.cat(
            [patterns[qkv_name] for qkv_name in qkv_names], dim=0
        )

    return state_dict


def convert_model(model_name: str, push_to_hub: bool = False):
    params = PRETRAINED_CHECKPOINTS[model_name]

    print(f"Converting model: {model_name}")
    print(f"Downloading weights from: {params['repo_id']}")

    hf_weights_path = hf_hub_download(
        repo_id=params["repo_id"], filename="pytorch_model.bin"
    )
    hf_state_dict = torch.load(hf_weights_path, weights_only=True)
    print(f"Loaded HuggingFace state dict with {len(hf_state_dict)} keys")

    # Rename keys
    keys_mapping = convert_old_keys_to_new_keys(hf_state_dict.keys(), params["mapping"])

    smp_state_dict = {}
    for old_key, new_key in keys_mapping.items():
        smp_state_dict[new_key] = hf_state_dict[old_key]

    # remove aux head
    smp_state_dict = {
        k: v for k, v in smp_state_dict.items() if "auxiliary_head." not in k
    }

    # [swin] group qkv layers and remove `relative_position_index`
    smp_state_dict = group_qkv_layers(smp_state_dict)
    smp_state_dict = {
        k: v for k, v in smp_state_dict.items() if "relative_position_index" not in k
    }

    # Create model
    print(f"Creating SMP UPerNet model with encoder: {params['encoder_name']}")
    extra_kwargs = params.get("extra_kwargs", {})
    smp_model = smp.UPerNet(
        encoder_name=params["encoder_name"],
        encoder_weights=None,
        decoder_channels=params["decoder_channels"],
        classes=params["classes"],
        **extra_kwargs,
    )

    print("Loading weights into SMP model...")
    smp_model.load_state_dict(smp_state_dict, strict=True)

    # Check we can run the model
    print("Verifying model with test inference...")
    smp_model.eval()
    sample = torch.ones(1, 3, 512, 512)
    with torch.inference_mode():
        output = smp_model(sample)
    print(f"Test inference successful. Output shape: {output.shape}")

    # Save model with preprocessing
    smp_repo_id = f"smp-hub/upernet-{model_name}"
    print(f"Saving model to: {smp_repo_id}")
    smp_model.save_pretrained(save_directory=smp_repo_id)

    transform = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
                max_pixel_value=1.0,
            ),
        ]
    )
    transform.save_pretrained(save_directory=smp_repo_id)

    if push_to_hub:
        print(f"Pushing model to HuggingFace Hub: {smp_repo_id}")
        api = HfApi()
        if not api.repo_exists(smp_repo_id):
            api.create_repo(repo_id=smp_repo_id, repo_type="model")
        api.upload_folder(
            repo_id=smp_repo_id,
            folder_path=smp_repo_id,
            repo_type="model",
        )

    print(f"Conversion of {model_name} completed successfully!")


if __name__ == "__main__":
    print(f"Starting conversion of {len(PRETRAINED_CHECKPOINTS)} UPerNet models")
    for model_name in PRETRAINED_CHECKPOINTS.keys():
        convert_model(model_name, push_to_hub=True)
    print("All conversions completed!")
