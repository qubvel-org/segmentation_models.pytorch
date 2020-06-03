import segmentation_models_pytorch as smp


def build_model(configuration):
    model_list = ['UNet', 'LinkNet', 'PSPNet', 'FPN', 'PAN', 'Deeplab_v3', 'Deeplab_v3+']
    if configuration.Model.model_name.lower() == 'unet':
        return smp.Unet(
            encoder_name=configuration.Model.encoder,
            encoder_weights=configuration.Model.encoder_weights,
            activation=None,
            classes=configuration.DataSet.number_of_classes,
            decoder_attention_type=None,
        )
    if configuration.Model.model_name.lower() == 'linknet':
        return smp.Linknet(
            encoder_name=configuration.Model.encoder,
            encoder_weights=configuration.Model.encoder_weights,
            activation=None,
            classes=configuration.DataSet.number_of_classes
        )
    if configuration.Model.model_name.lower() == 'pspnet':
        return smp.PSPNet(
            encoder_name=configuration.Model.encoder,
            encoder_weights=configuration.Model.encoder_weights,
            activation=None,
            classes=configuration.DataSet.number_of_classes
        )
    if configuration.Model.model_name.lower() == 'fpn':
        return smp.FPN(
            encoder_name=configuration.Model.encoder,
            encoder_weights=configuration.Model.encoder_weights,
            activation=None,
            classes=configuration.DataSet.number_of_classes
        )
    if configuration.Model.model_name.lower() == 'pan':
        return smp.PAN(
            encoder_name=configuration.Model.encoder,
            encoder_weights=configuration.Model.encoder_weights,
            activation=None,
            classes=configuration.DataSet.number_of_classes
        )
    if configuration.Model.model_name.lower() == 'deeplab_v3+':
        return smp.DeepLabV3Plus(
            encoder_name=configuration.Model.encoder,
            encoder_weights=configuration.Model.encoder_weights,
            activation=None,
            classes=configuration.DataSet.number_of_classes
        )
    if configuration.Model.model_name.lower() == 'deeplab_v3':
        return smp.DeepLabV3(
            encoder_name=configuration.Model.encoder,
            encoder_weights=configuration.Model.encoder_weights,
            activation=None,
            classes=configuration.DataSet.number_of_classes
        )
    raise KeyError(f'Model should be one of {model_list}')
