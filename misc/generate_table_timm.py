import timm
from tqdm import tqdm


def check_features_and_reduction(name):
    encoder = timm.create_model(name, features_only=True, pretrained=False)
    if not encoder.feature_info.reduction() == [2, 4, 8, 16, 32]:
        raise ValueError


def has_dilation_support(name):
    try:
        timm.create_model(name, features_only=True, output_stride=8, pretrained=False)
        timm.create_model(name, features_only=True, output_stride=16, pretrained=False)
        return True
    except Exception:
        return False


def make_table(data):
    names = data.keys()
    max_len1 = max([len(x) for x in names]) + 2
    max_len2 = len("support dilation") + 2

    l1 = "+" + "-" * max_len1 + "+" + "-" * max_len2 + "+\n"
    l2 = "+" + "=" * max_len1 + "+" + "=" * max_len2 + "+\n"
    top = "| " + "Encoder name".ljust(max_len1 - 2) + " | " + "Support dilation".center(max_len2 - 2) + " |\n"

    table = l1 + top + l2

    for k in sorted(data.keys()):
        support = "✅".center(max_len2 - 3) if data[k]["has_dilation"] else " ".center(max_len2 - 2)
        table += "| " + k.ljust(max_len1 - 2) + " | " + support + " |\n"
        table += l1

    return table


if __name__ == "__main__":

    supported_models = {}

    with tqdm(timm.list_models()) as names:
        for name in names:
            try:
                check_features_and_reduction(name)
                has_dilation = has_dilation_support(name)
                supported_models[name] = dict(has_dilation=has_dilation)
            except Exception:
                continue

    table = make_table(supported_models)
    print(table)
    with open("timm_encoders.txt", "w") as f:
        print(table, file=f)
    print(f"Total encoders: {len(supported_models.keys())}")
