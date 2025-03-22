import segmentation_models_pytorch as smp
import torch
import huggingface_hub

MODEL_WEIGHTS_PATH = r"C:\Users\vedan\Downloads\dpt_large-ade20k-b12dca68.pt"
HF_HUB_PATH = "vedantdalimkar/DPT"

if __name__ == "__main__":
    smp_model = smp.DPT(encoder_name="tu-vit_large_patch16_384", classes=150)
    dpt_model_dict = torch.load(MODEL_WEIGHTS_PATH)

    for layer_index in range(0, 4):
        for param in [
            "running_mean",
            "running_var",
            "num_batches_tracked",
            "weight",
            "bias",
        ]:
            for block_index in [1, 2]:
                for bn_index in [1, 2]:
                    # Assigning weights of 4th fusion layer of original model to 1st layer of SMP DPT model,
                    # Assigning weights of 3rd fusion layer of original model to 2nd layer of SMP DPT model ...
                    # and so on ...

                    # This is because order of calling fusion layers is reversed in original DPT implementation

                    dpt_model_dict[
                        f"decoder.fusion_blocks.{layer_index}.residual_conv_block{block_index}.batch_norm_{bn_index}.{param}"
                    ] = dpt_model_dict.pop(
                        f"scratch.refinenet{4 - layer_index}.resConfUnit{block_index}.bn{bn_index}.{param}"
                    )

            if param in ["weight", "bias"]:
                if param == "weight":
                    for block_index in [1, 2]:
                        for conv_index in [1, 2]:
                            dpt_model_dict[
                                f"decoder.fusion_blocks.{layer_index}.residual_conv_block{block_index}.conv_{conv_index}.{param}"
                            ] = dpt_model_dict.pop(
                                f"scratch.refinenet{4 - layer_index}.resConfUnit{block_index}.conv{conv_index}.{param}"
                            )

                    dpt_model_dict[
                        f"decoder.reassemble_blocks.{layer_index}.project_to_feature_dim.{param}"
                    ] = dpt_model_dict.pop(f"scratch.layer{layer_index + 1}_rn.{param}")

                dpt_model_dict[
                    f"decoder.fusion_blocks.{layer_index}.project.{param}"
                ] = dpt_model_dict.pop(
                    f"scratch.refinenet{4 - layer_index}.out_conv.{param}"
                )

                dpt_model_dict[
                    f"decoder.readout_blocks.{layer_index}.project.0.{param}"
                ] = dpt_model_dict.pop(
                    f"pretrained.act_postprocess{layer_index + 1}.0.project.0.{param}"
                )

                dpt_model_dict[
                    f"decoder.reassemble_blocks.{layer_index}.project_to_out_channel.{param}"
                ] = dpt_model_dict.pop(
                    f"pretrained.act_postprocess{layer_index + 1}.3.{param}"
                )

                if layer_index != 2:
                    dpt_model_dict[
                        f"decoder.reassemble_blocks.{layer_index}.upsample.{param}"
                    ] = dpt_model_dict.pop(
                        f"pretrained.act_postprocess{layer_index + 1}.4.{param}"
                    )

    # Changing state dict keys for segmentation head
    dpt_model_dict = {
        (
            "segmentation_head.head" + name[len("scratch.output_conv") :]
            if name.startswith("scratch.output_conv")
            else name
        ): parameter
        for name, parameter in dpt_model_dict.items()
    }

    # Changing state dict keys for encoder layers
    dpt_model_dict = {
        (
            "encoder.model" + name[len("pretrained.model") :]
            if name.startswith("pretrained.model")
            else name
        ): parameter
        for name, parameter in dpt_model_dict.items()
    }

    # Removing keys,value pairs associated with auxiliary head
    dpt_model_dict = {
        name: parameter
        for name, parameter in dpt_model_dict.items()
        if not name.startswith("auxlayer")
    }

    smp_model.load_state_dict(dpt_model_dict, strict=True)

    model_name = MODEL_WEIGHTS_PATH.split("\\")[-1].replace(".pt", "")

    smp_model.save_pretrained(model_name)

    repo_id = HF_HUB_PATH
    api = huggingface_hub.HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model")
    api.upload_folder(folder_path=model_name, repo_id=repo_id)
