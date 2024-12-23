import os
import torch
import tempfile
import huggingface_hub
import segmentation_models_pytorch as smp

HUB_REPO = "smp-test-models"
ENCODER_NAME = "tu-resnet18"

api = huggingface_hub.HfApi(token=os.getenv("HF_TOKEN"))

for model_name, model_class in smp.MODEL_ARCHITECTURES_MAPPING.items():
    model = model_class(encoder_name=ENCODER_NAME)
    model = model.eval()

    # generate test sample
    torch.manual_seed(423553)
    sample = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        output = model(sample)

    with tempfile.TemporaryDirectory() as tmpdir:
        # save model
        model.save_pretrained(f"{tmpdir}")

        # save input and output
        torch.save(sample, f"{tmpdir}/input-tensor.pth")
        torch.save(output, f"{tmpdir}/output-tensor.pth")

        # create repo
        repo_id = f"{HUB_REPO}/{model_name}-{ENCODER_NAME}"
        if not api.repo_exists(repo_id=repo_id):
            api.create_repo(repo_id=repo_id, repo_type="model")

        # upload to hub
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=f"{HUB_REPO}/{model_name}-{ENCODER_NAME}",
            repo_type="model",
        )
