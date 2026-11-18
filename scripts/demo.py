import json
import shutil
from io import BytesIO
from pathlib import Path
from tempfile import SpooledTemporaryFile, TemporaryDirectory
from zipfile import ZipFile

import torch

from scripts.evaluate_holdout import read_holdout
from scripts.image_data import load_image
from scripts.train_resnet import make_model as make_resnet
from scripts.train_vit import make_model as make_vit


def read_case(holdout, image_id=None):
    with ZipFile(holdout) as archive:
        labels = read_holdout(archive)
        ids = sorted(labels)
        if image_id is None:
            image_id = ids[len(ids) // 2]
        if image_id not in labels:
            raise ValueError("Choose an image ID listed in the RSNA holdout labels")

        for entry in archive.infolist():
            if not entry.filename.lower().endswith(".zip"):
                continue
            with archive.open(entry) as source, SpooledTemporaryFile(max_size=8_000_000) as nested:
                shutil.copyfileobj(source, nested)
                with ZipFile(nested) as images:
                    for member in images.infolist():
                        if Path(member.filename).name == f"{image_id}.png":
                            return image_id, labels[image_id][0], images.read(member)

    raise ValueError(f"Could not find image {image_id} in the RSNA holdout")


def read_run(path, name):
    with ZipFile(path) as archive:
        info = json.loads(archive.read("run.json"))
        if info["model"] != name or info["weights"] != "IMAGENET1K_V1":
            raise ValueError(f"Expected a {name} ImageNet V1 run in {path}")
        head = torch.load(BytesIO(archive.read("best-head.pt")), map_location="cpu", weights_only=True)
    return info, head


@torch.inference_mode()
def demo(holdout, resnet_zip, vit_zip, image_id=None):
    resnet, resnet_head = read_run(resnet_zip, "resnet18")
    vit, vit_head = read_run(vit_zip, "vit_b_16")
    for key in ("labels_sha256", "split_sha256"):
        if resnet[key] != vit[key]:
            raise ValueError("The two checkpoints were trained on different data or splits")

    image_id, reference, png = read_case(holdout, image_id)
    with TemporaryDirectory() as temp:
        path = Path(temp) / "case.png"
        path.write_bytes(png)
        image = load_image(path).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = (
        ("ResNet-18", make_resnet, "fc", resnet_head),
        ("ViT-B/16", make_vit, "heads.head", vit_head),
    )
    estimates = {}
    for name, make_model, layer, head in models:
        model = make_model(0)
        model.get_submodule(layer).load_state_dict(head)
        estimates[name] = model.to(device)(image.to(device)).item()
        del model
    return image_id, reference, png, estimates
