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


def valid_vit_encoder_for_dpt(name):
    if "vit" not in name:
        return False
    encoder = timm.create_model(name)
    feature_info = encoder.feature_info
    feature_info_obj = timm.models.FeatureInfo(
        feature_info=feature_info, out_indices=[0, 1, 2, 3]
    )
    reduction_scales = list(feature_info_obj.reduction())

    if len(set(reduction_scales)) > 1:
        return False

    output_stride = reduction_scales[0]
    if bin(output_stride).count("1") != 1:
        return False

    return True


def make_table(data):
    names = data.keys()
    max_len1 = max([len(x) for x in names]) + 2
    max_len2 = len("support dilation") + 2
    max_len3 = len("Supported for DPT") + 2

    l1 = "+" + "-" * max_len1 + "+" + "-" * max_len2 + "+" + "-" * max_len3 + "+\n"
    l2 = "+" + "=" * max_len1 + "+" + "=" * max_len2 + "+" + "-" * max_len3 + "+\n"
    top = (
        "| "
        + "Encoder name".ljust(max_len1 - 2)
        + " | "
        + "Support dilation".center(max_len2 - 2)
        + " | "
        + "Supported for DPT".center(max_len3 - 2)
        + " |\n"
    )

    table = l1 + top + l2

    for k in sorted(data.keys()):
        if "has_dilation" in data[k] and data[k]["has_dilation"]:
            support = "✅".center(max_len2 - 3)

        else:
            support = " ".center(max_len2 - 2)

        if "supported_only_for_dpt" in data[k]:
            supported_for_dpt = "✅".center(max_len3 - 3)

        else:
            supported_for_dpt = " ".center(max_len3 - 2)

        table += (
            "| "
            + k.ljust(max_len1 - 2)
            + " | "
            + support
            + " | "
            + supported_for_dpt
            + " |\n"
        )
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
                try:
                    if valid_vit_encoder_for_dpt(name):
                        supported_models[name] = dict(supported_only_for_dpt=True)
                except Exception:
                    continue

    table = make_table(supported_models)
    print(table)
    with open("timm_encoders.txt", "w") as f:
        print(table, file=f)
    print(f"Total encoders: {len(supported_models.keys())}")
