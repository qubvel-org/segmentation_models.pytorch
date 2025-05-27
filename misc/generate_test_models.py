import os
import torch
import tempfile
import huggingface_hub
import segmentation_models_pytorch as smp

HUB_REPO = "smp-test-models"
ENCODER_NAME = "tu-resnet18"

api = huggingface_hub.HfApi(token=os.getenv("HF_TOKEN"))


def save_and_push(model, inputs, outputs, model_name, encoder_name):
    with tempfile.TemporaryDirectory() as tmpdir:
        # save model
        model.save_pretrained(f"{tmpdir}")

        # save input and output
        torch.save(inputs, f"{tmpdir}/input-tensor.pth")
        torch.save(outputs, f"{tmpdir}/output-tensor.pth")

        # create repo
        repo_id = f"{HUB_REPO}/{model_name}-{encoder_name}"
        if not api.repo_exists(repo_id=repo_id):
            api.create_repo(repo_id=repo_id, repo_type="model")

        # upload to hub
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=f"{HUB_REPO}/{model_name}-{encoder_name}",
            repo_type="model",
        )


for model_name, model_class in smp.MODEL_ARCHITECTURES_MAPPING.items():
    if model_name == "dpt":
        encoder_name = "tu-test_vit"
        model = smp.DPT(
            encoder_name=encoder_name,
            decoder_readout="cat",
            decoder_intermediate_channels=(16, 32, 64, 64),
            decoder_fusion_channels=16,
            dynamic_img_size=True,
        )
    else:
        encoder_name = ENCODER_NAME
        model = model_class(encoder_name=encoder_name)

    model = model.eval()

    # generate test sample
    torch.manual_seed(423553)
    sample = torch.rand(1, 3, 256, 256)

    with torch.inference_mode():
        output = model(sample)

    save_and_push(model, sample, output, model_name, encoder_name)
