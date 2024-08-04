import random

import blobfile as bf
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _list_image_files_recursively(data_dir):  # TODO re-read and re-write
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def center_crop_arr(pil_image, image_size):  # TODO re-read and re-write:
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class ImageDataset(Dataset):
    def __init__(self, resolution, images_paths, classes=None):

        super().__init__()
        self.resolution = resolution
        self.local_images = images_paths
        self.classes: int = classes
        self.random_flip = False

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):  # TODO re-read and re-write:
        path: strt = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        arr = center_crop_arr(pil_image, self.resolution)
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.classes is not None:
            out_dict["y"] = np.array(self.classes[idx], dtype=np.int64)
        if len(arr.shape) == 2:
            arr = np.stack((arr,) * 3, axis=-1)
        return np.transpose(arr, [2, 0, 1]), out_dict


def load_data(
    data_dir: str,
    batch_size: int,
    image_size: int,
    shuffle: bool = False,
    class_cond: bool = False,
):
    classes = None  # TODO to be updated after implementing loading data without labels
    all_files = _list_image_files_recursively(data_dir)
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = ImageDataset(image_size, all_files, classes=classes)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )
    yield from loader
