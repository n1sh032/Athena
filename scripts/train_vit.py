from torchvision.models import ViT_B_16_Weights, vit_b_16

from scripts.train_head import main, make_head


def make_model(start_age, weights=ViT_B_16_Weights.IMAGENET1K_V1):
    model = vit_b_16(weights=weights)
    model.requires_grad_(False)
    model.heads.head = make_head(model.heads.head.in_features, start_age)
    return model.eval()


if __name__ == "__main__":
    main("vit_b_16", make_model, "heads.head")
