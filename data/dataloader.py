import blobfile as bf
from torch.utils.data import DataLoader, Dataset

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def load_data(
        data_dir: str,
        batch_size: int,
        image_size: int,
        class_cond: bool = False
):
    classes = None # TODO to be updated after implementing loading data without labels
    dataste = ImageDataset(
        image_size,
        all_files,
        classes=classes
    )
    loader = DataLoader (
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )        )
    