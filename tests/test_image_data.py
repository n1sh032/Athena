import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from torch.utils.data import DataLoader

from scripts.image_data import load_image, make_datasets, read_split


class ImageDataTest(unittest.TestCase):
    def test_batch_keeps_labels_and_pads_without_cropping(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            labels = root / "train.csv"
            labels.write_text("id,boneage,male\n1,80,True\n2,120,False\n3,180,True\n")
            split = root / "split.json"
            split.write_text(json.dumps({"train_ids": [3, 1], "dev_ids": [2]}))
            for image_id in (1, 2, 3):
                Image.new("L", (200, 100), 128).save(root / f"{image_id}.png")

            train, dev = make_datasets(labels, root, split)
            images, ages = next(iter(DataLoader(train, batch_size=2)))
            self.assertEqual(tuple(images.shape), (2, 3, 224, 224))
            self.assertEqual(images.dtype, torch.float32)
            self.assertEqual(ages.tolist(), [180, 80])
            self.assertEqual(dev[0][1].item(), 120)
            self.assertAlmostEqual(images[0, 0, 0, 0].item(), -0.485 / 0.229, places=5)
            self.assertAlmostEqual(
                images[0, 0, 112, 0].item(), (128 / 255 - 0.485) / 0.229, places=5
            )

            (root / "2.png").unlink()
            with self.assertRaisesRegex(ValueError, "Missing PNGs: 1"):
                make_datasets(labels, root, split)

    def test_split_rejects_overlap(self):
        with TemporaryDirectory() as folder:
            split = Path(folder) / "split.json"
            split.write_text(json.dumps({"train_ids": [1, 2], "dev_ids": [2]}))
            with self.assertRaisesRegex(ValueError, "duplicate IDs"):
                read_split(split, {1: (80, True), 2: (120, False)})

    def test_png_decode(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "hand.png"
            Image.new("I;16", (16, 16), 32896).save(path)
            pixel = load_image(path)[0, 112, 112].item()
            self.assertAlmostEqual(pixel, (128 / 255 - 0.485) / 0.229, places=5)

            path.write_bytes(b"broken image")
            with self.assertRaisesRegex(ValueError, "Could not load"):
                load_image(path)
