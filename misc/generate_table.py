import os
import segmentation_models_pytorch as smp

from tqdm import tqdm

encoders = smp.encoders.encoders


WIDTH = 32
COLUMNS = ["Encoder", "Pretrained weights", "Params, M", "Script", "Compile", "Export"]
FILE = "encoders_table.md"

if os.path.exists(FILE):
    os.remove(FILE)


def wrap_row(r):
    return "|{}|".format(r)


header = "|".join([column.ljust(WIDTH, " ") for column in COLUMNS])
separator = "|".join(
    ["-" * WIDTH] + [":" + "-" * (WIDTH - 2) + ":"] * (len(COLUMNS) - 1)
)

print(wrap_row(header), file=open(FILE, "a"))
print(wrap_row(separator), file=open(FILE, "a"))

for encoder_name, encoder in tqdm(encoders.items()):
    weights = "<br>".join(encoder["pretrained_settings"].keys())

    model = encoder["encoder"](**encoder["params"], depth=5)

    script = "✅" if model._is_torch_scriptable else "❌"
    compile = "✅" if model._is_torch_compilable else "❌"
    export = "✅" if model._is_torch_exportable else "❌"

    params = sum(p.numel() for p in model.parameters())
    params = str(params // 1000000) + "M"

    row = [encoder_name, weights, params, script, compile, export]
    row = [str(r).ljust(WIDTH, " ") for r in row]
    row = "|".join(row)

    print(wrap_row(row), file=open(FILE, "a"))
