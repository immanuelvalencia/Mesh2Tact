"""Torchvision classifier families supported by training and Predict."""

SUITES = {
    "resnet": ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"),
    "efficientnet": ("efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
                     "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
                     "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"),
    "densenet": ("densenet121", "densenet161", "densenet169", "densenet201"),
    "mobilenet": ("mobilenet_v3_small", "mobilenet_v3_large"),
    "convnext": ("convnext_tiny", "convnext_small", "convnext_base", "convnext_large"),
    "swin": ("swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b"),
    "vit": ("vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"),
    "regnet": ("regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf",
               "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf"),
}

ALL_MODELS = tuple(dict.fromkeys(name for family in SUITES.values() for name in family))
