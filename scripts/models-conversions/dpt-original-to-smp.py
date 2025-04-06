import cv2
import torch
import albumentations as A
import segmentation_models_pytorch as smp

MODEL_WEIGHTS_PATH = r"dpt_large-ade20k-b12dca68.pt"
HF_HUB_PATH = "qubvel-hf/dpt-large-ade20k"
PUSH_TO_HUB = False


def get_transform():
    return A.Compose(
        [
            A.LongestMaxSize(max_size=480, interpolation=cv2.INTER_CUBIC),
            A.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0
            ),
            # This is not correct transform, ideally image should resized without padding to multiple of 32,
            # but we take there is no such transform in albumentations, here is closest one
            A.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=32,
                pad_width_divisor=32,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1,
            ),
        ]
    )


if __name__ == "__main__":
    # fmt: off
    smp_model = smp.DPT(encoder_name="tu-vit_large_patch16_384", classes=150, dynamic_img_size=True)
    dpt_model_dict = torch.load(MODEL_WEIGHTS_PATH, weights_only=True)

    for layer_index in range(0, 4):
        for param in ["running_mean", "running_var", "num_batches_tracked", "weight", "bias"]:
            for block_index in [1, 2]:
                for bn_index in [1, 2]:
                    # Assigning weights of 4th fusion layer of original model to 1st layer of SMP DPT model,
                    # Assigning weights of 3rd fusion layer of original model to 2nd layer of SMP DPT model ...
                    # and so on ...
                    # This is because order of calling fusion layers is reversed in original DPT implementation
                    dpt_model_dict[f"decoder.fusion_blocks.{layer_index}.residual_conv_block{block_index}.batch_norm_{bn_index}.{param}"] = \
                        dpt_model_dict.pop(f"scratch.refinenet{4 - layer_index}.resConfUnit{block_index}.bn{bn_index}.{param}")

            if param in ["weight", "bias"]:
                if param == "weight":
                    for block_index in [1, 2]:
                        for conv_index in [1, 2]:
                            dpt_model_dict[f"decoder.fusion_blocks.{layer_index}.residual_conv_block{block_index}.conv_{conv_index}.{param}"] = \
                                dpt_model_dict.pop(f"scratch.refinenet{4 - layer_index}.resConfUnit{block_index}.conv{conv_index}.{param}")

                    dpt_model_dict[f"decoder.reassemble_blocks.{layer_index}.project_to_feature_dim.{param}"] = \
                        dpt_model_dict.pop(f"scratch.layer{layer_index + 1}_rn.{param}")

                dpt_model_dict[f"decoder.fusion_blocks.{layer_index}.project.{param}"] = \
                    dpt_model_dict.pop(f"scratch.refinenet{4 - layer_index}.out_conv.{param}")

                dpt_model_dict[f"decoder.projection_blocks.{layer_index}.project.0.{param}"] = \
                    dpt_model_dict.pop(f"pretrained.act_postprocess{layer_index + 1}.0.project.0.{param}")

                dpt_model_dict[f"decoder.reassemble_blocks.{layer_index}.project_to_out_channel.{param}"] = \
                    dpt_model_dict.pop(f"pretrained.act_postprocess{layer_index + 1}.3.{param}")

                if layer_index != 2:
                    dpt_model_dict[f"decoder.reassemble_blocks.{layer_index}.upsample.{param}"] = \
                        dpt_model_dict.pop(f"pretrained.act_postprocess{layer_index + 1}.4.{param}")

    # Changing state dict keys for segmentation head
    dpt_model_dict = {
        name.replace("scratch.output_conv", "segmentation_head.head"): parameter
        for name, parameter in dpt_model_dict.items()
    }

    # Changing state dict keys for encoder layers
    dpt_model_dict = {
        name.replace("pretrained.model", "encoder.model"): parameter
        for name, parameter in dpt_model_dict.items()
    }

    # Removing keys, value pairs associated with auxiliary head
    dpt_model_dict = {
        name: parameter
        for name, parameter in dpt_model_dict.items()
        if not name.startswith("auxlayer")
    }
    # fmt: on

    smp_model.load_state_dict(dpt_model_dict, strict=True)

    # ------- DO NOT touch this section -------
    smp_model.eval()

    input_tensor = torch.ones((1, 3, 384, 384))
    output = smp_model(input_tensor)

    print(output.shape)
    print(output[0, 0, :3, :3])

    expected_slice = torch.tensor(
        [
            [3.4243, 3.4553, 3.4863],
            [3.3332, 3.2876, 3.2419],
            [3.2422, 3.1199, 2.9975],
        ]
    )

    torch.testing.assert_close(
        output[0, 0, :3, :3], expected_slice, atol=1e-4, rtol=1e-4
    )

    # Saving
    transform = get_transform()

    transform.save_pretrained(HF_HUB_PATH)
    smp_model.save_pretrained(HF_HUB_PATH, push_to_hub=PUSH_TO_HUB)

    # Re-loading to make sure everything is saved correctly
    smp_model = smp.from_pretrained(HF_HUB_PATH)
