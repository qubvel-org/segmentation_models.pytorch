import torch
import argparse
import requests
import numpy as np
import huggingface_hub
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image
import segmentation_models_pytorch as smp


def convert_state_dict_to_smp(state_dict: dict):
    # fmt: off

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = {}

    # Map the backbone components to the encoder
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith("backbone"):
            new_key = key.replace("backbone", "encoder")
            new_state_dict[new_key] = state_dict.pop(key)


    # Map the linear_cX layers to MLP stages
    for i in range(4):
        base = f"decode_head.linear_c{i+1}.proj"
        new_state_dict[f"decoder.mlp_stage.{3-i}.linear.weight"] = state_dict.pop(f"{base}.weight")
        new_state_dict[f"decoder.mlp_stage.{3-i}.linear.bias"] = state_dict.pop(f"{base}.bias")

    # Map fuse_stage components
    fuse_base = "decode_head.linear_fuse"
    fuse_weights = {
        "decoder.fuse_stage.0.weight": state_dict.pop(f"{fuse_base}.conv.weight"),
        "decoder.fuse_stage.1.weight": state_dict.pop(f"{fuse_base}.bn.weight"),
        "decoder.fuse_stage.1.bias": state_dict.pop(f"{fuse_base}.bn.bias"),
        "decoder.fuse_stage.1.running_mean": state_dict.pop(f"{fuse_base}.bn.running_mean"),
        "decoder.fuse_stage.1.running_var": state_dict.pop(f"{fuse_base}.bn.running_var"),
        "decoder.fuse_stage.1.num_batches_tracked": state_dict.pop(f"{fuse_base}.bn.num_batches_tracked"),
    }
    new_state_dict.update(fuse_weights)

    # Map final layer components
    new_state_dict["segmentation_head.0.weight"] = state_dict.pop("decode_head.linear_pred.weight")
    new_state_dict["segmentation_head.0.bias"] = state_dict.pop("decode_head.linear_pred.bias")

    del state_dict["decode_head.conv_seg.weight"]
    del state_dict["decode_head.conv_seg.bias"]

    assert len(state_dict) == 0, f"Unmapped keys: {state_dict.keys()}"

    # fmt: on
    return new_state_dict


def get_np_image():
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return np.array(image)


def main(args):
    original_checkpoint = torch.load(args.path, map_location="cpu", weights_only=True)
    smp_state_dict = convert_state_dict_to_smp(original_checkpoint)

    config = original_checkpoint["meta"]["config"]
    num_classes = int(config.split("num_classes=")[1].split(",\n")[0])
    decoder_dims = int(config.split("embed_dim=")[1].split(",\n")[0])
    height, width = [
        int(x) for x in config.split("crop_size=(")[1].split("), ")[0].split(",")
    ]
    model_size = args.path.split("segformer.")[1][:2]

    # Create the model
    model = smp.create_model(
        in_channels=3,
        classes=num_classes,
        arch="segformer",
        encoder_name=f"mit_{model_size}",
        encoder_weights=None,
        decoder_segmentation_channels=decoder_dims,
    ).eval()

    # Load the converted state dict
    model.load_state_dict(smp_state_dict, strict=True)

    # Preprocessing params
    preprocessing = A.Compose(
        [
            A.Resize(height, width, p=1),
            A.Normalize(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                max_pixel_value=1.0,
                p=1,
            ),
        ]
    )

    # Prepare the input
    image = get_np_image()
    normalized_image = preprocessing(image=image)["image"]
    tensor = torch.tensor(normalized_image).permute(2, 0, 1).unsqueeze(0).float()

    # Forward pass
    with torch.no_grad():
        mask = model(tensor)

    # Postprocessing
    mask = torch.nn.functional.interpolate(
        mask, size=(image.shape[0], image.shape[1]), mode="bilinear"
    )
    mask = torch.argmax(mask, dim=1)
    mask = mask.squeeze().cpu().numpy()

    model_name = args.path.split("/")[-1].replace(".pth", "").replace(".", "-")

    model.save_pretrained(model_name)
    preprocessing.save_pretrained(model_name)

    # fmt: off
    plt.subplot(121), plt.axis('off'), plt.imshow(image), plt.title('Input Image')
    plt.subplot(122), plt.axis('off'), plt.imshow(mask), plt.title('Output Mask')
    plt.savefig(f"{model_name}/example_mask.png")
    # fmt: on

    if args.push_to_hub:
        repo_id = f"smp-hub/{model_name}"
        api = huggingface_hub.HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model")
        api.upload_folder(folder_path=model_name, repo_id=repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="weights/trained_models/segformer.b2.512x512.ade.160k.pth",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    main(args)
