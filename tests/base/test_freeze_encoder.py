import torch
import segmentation_models_pytorch as smp


def test_freeze_and_unfreeze_encoder():
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None)

    def assert_encoder_params_trainable(expected: bool):
        assert all(p.requires_grad == expected for p in model.encoder.parameters())

    def assert_norm_layers_training(expected: bool):
        for m in model.encoder.modules():
            if isinstance(m, torch.nn.modules.batchnorm._NormBase):
                assert m.training == expected

    # Initially, encoder params should be trainable
    model.train()
    assert_encoder_params_trainable(True)

    # Freeze encoder
    model.freeze_encoder()
    assert_encoder_params_trainable(False)
    assert_norm_layers_training(False)

    # Call train() and ensure encoder norm layers stay frozen
    model.train()
    assert_norm_layers_training(False)

    # Unfreeze encoder
    model.unfreeze_encoder()
    assert_encoder_params_trainable(True)
    assert_norm_layers_training(True)

    # Call train() again — should stay trainable
    model.train()
    assert_norm_layers_training(True)

    # Switch to eval, then freeze
    model.eval()
    model.freeze_encoder()
    assert_encoder_params_trainable(False)
    assert_norm_layers_training(False)

    # Unfreeze again
    model.unfreeze_encoder()
    assert_encoder_params_trainable(True)
    assert_norm_layers_training(True)


def test_freeze_encoder_stops_running_stats():
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None)
    model.freeze_encoder()
    model.train()  # overridden train, encoder should remain frozen
    bn = None

    for m in model.encoder.modules():
        if isinstance(m, torch.nn.modules.batchnorm._NormBase):
            bn = m
            break

    assert bn is not None

    orig_mean = bn.running_mean.clone()
    orig_var = bn.running_var.clone()

    x = torch.randn(2, 3, 64, 64)
    _ = model(x)

    torch.testing.assert_close(orig_mean, bn.running_mean)
    torch.testing.assert_close(orig_var, bn.running_var)
