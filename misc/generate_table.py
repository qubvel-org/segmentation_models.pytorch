import segmentation_models_pytorch as smp

encoders = smp.encoders.encoders


WIDTH = 32
COLUMNS = [
    "Encoder",
    "Weights",
    "Params, M",
]


def wrap_row(r):
    return "|{}|".format(r)


header = "|".join([column.ljust(WIDTH, " ") for column in COLUMNS])
separator = "|".join(["-" * WIDTH] + [":" + "-" * (WIDTH - 2) + ":"] * (len(COLUMNS) - 1))

print(wrap_row(header))
print(wrap_row(separator))

for encoder_name, encoder in encoders.items():
    weights = "<br>".join(encoder["pretrained_settings"].keys())
    encoder_name = encoder_name.ljust(WIDTH, " ")
    weights = weights.ljust(WIDTH, " ")

    model = encoder["encoder"](**encoder["params"], depth=5)
    params = sum(p.numel() for p in model.parameters())
    params = str(params // 1000000) + "M"
    params = params.ljust(WIDTH, " ")

    row = "|".join([encoder_name, weights, params])
    print(wrap_row(row))
