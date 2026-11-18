from torchvision.models import ResNet18_Weights, resnet18

from scripts.train_head import main, make_head


def make_model(start_age, weights=ResNet18_Weights.IMAGENET1K_V1):
    model = resnet18(weights=weights)
    model.requires_grad_(False)
    model.fc = make_head(model.fc.in_features, start_age)
    # Evaluation mode keeps the frozen batch norm statistics fixed.
    return model.eval()


if __name__ == "__main__":
    main("resnet18", make_model, "fc")
