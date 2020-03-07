hrnet_config = {
    "MODEL": {
        "NAME": 'HRNet',
        "PRETRAINED": '',
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [2],
                "NUM_CHANNELS": [16],
                "FUSE_METHOD": "SUM"
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2,2],
                "NUM_CHANNELS": [16,32],
                "FUSE_METHOD": "SUM"
            },
            "STAGE3": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2,2,2],
                "NUM_CHANNELS": [16,32,64],
                "FUSE_METHOD": "SUM"
            },
            "STAGE4": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2,2,2,2],
                "NUM_CHANNELS": [16,32,64,128],
                "FUSE_METHOD": "SUM",
            }
        }
    }
}

