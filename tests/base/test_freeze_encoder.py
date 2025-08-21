import torch
import segmentation_models_pytorch as smp


def test_freeze_and_unfreeze_encoder():
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None)
    
    # Initially, encoder params should be trainable
    model.train()
    assert all(p.requires_grad for p in model.encoder.parameters())
    
    # Check encoder params are frozen
    model.freeze_encoder()
    
    assert all(not p.requires_grad for p in model.encoder.parameters())
    for m in model.encoder.modules():
        if isinstance(m, torch.nn.modules.batchnorm._NormBase):
            assert not m.training

    # Call train() and ensure encoder norm layers stay frozen
    model.train()
    for m in model.encoder.modules():
        if isinstance(m, torch.nn.modules.batchnorm._NormBase):
            assert not m.training
    
    # Params should be trainable again
    model.unfreeze_encoder()
    
    assert all(p.requires_grad for p in model.encoder.parameters())
    # Norm layers should go back to training mode after unfreeze
    for m in model.encoder.modules():
        if isinstance(m, torch.nn.modules.batchnorm._NormBase):
            assert m.training
    
    model.train()
    # Norm layers should have the same training mode after train()
    for m in model.encoder.modules():
        if isinstance(m, torch.nn.modules.batchnorm._NormBase):
            assert m.training


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
