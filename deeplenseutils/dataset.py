import os

import gdown
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

from configs.data_config import DATASET
from deeplenseutils.augmentation import DefaultTransformations
from deeplenseutils.util import make_directories


def download_dataset(
    filename: str,
    url: str = "https://drive.google.com/uc?id=1m7QzSzXyE8u_QoYplN9dIe-X2pf1KXxt",
) -> None:
    """Downloads dataset from Google drive

    Args:
        filename (str): output directory name
        url (str, optional): URL to dataset. Defaults to "https://drive.google.com/uc?id=1m7QzSzXyE8u_QoYplN9dIe-X2pf1KXxt".
    """
    if not os.path.isfile(filename):
        try:
            gdown.download(url, filename, quiet=False)
        except Exception as e:
            print(e)
    else:
        print("File exists")


def extract_split_dataset(
    filename: str,
    destination_dir: str = "data",
    dataset_name: str = "Model_I",
    split: bool = False,
) -> None:
    """Extract from .tar file and splits dataset (90:10) into train and validation set

    Args:
        filename (str): tar filename
        destination_dir (str, optional): output directory name. Defaults to "data".
        dataset_name (str, optional): dataset name: Model_I, Model_II, Model_III. Defaults to "Model_I".
        split (bool, optional): whether to split or not. Defaults to False.
    """
    # only extracting folder
    if not split:
        print("Extracting folder ...")
        os.system(f"tar xf {filename} --directory {destination_dir}")
        print("Extraction complete")
        # os.system(f"rm -r {filename}")

    # splitting folder
    else:
        os.system(
            f"tar xf {filename} --directory {destination_dir} ; mv {destination_dir}/{dataset_name} {destination_dir}/{dataset_name}_raw"
        )
        splitfolders.ratio(
            f"{destination_dir}/{dataset_name}_raw",
            output=f"{destination_dir}/{dataset_name}",
            seed=1337,
            ratio=(0.9, 0.1),
        )
        os.system(f"rm -r {destination_dir}/{dataset_name}_raw")


class DeepLenseDataset(Dataset):
    def __init__(
        self,
        destination_dir: str,
        mode: str,
        dataset_name: str,
        transform=None,
        download="False",
        channels=1,
    ):
        """Class for DeepLense dataset

        Args:
            destination_dir (str): directory where dataset is stored \n
            mode (str): type of dataset:  `train` or `val` or `test`
            dataset_name (str): name of dataset e.g. Model_I
            transform (_type_, optional): transformation of images. Defaults to None.
            download (str, optional): whether to download the dataset. Defaults to "False".
            channels (int, optional): # of channels. Defaults to 1.

        Example:
            >>>     trainset = DeepLenseDataset(
            >>>     dataset_dir,
            >>>     "train",
            >>>     dataset_name,
            >>>     transform=get_transform_train(
            >>>     upsample_size=387,
            >>>     final_size=train_config["image_size"],
            >>>     channels=train_config["channels"]),
            >>>     download=True,
            >>>     channels=train_config["channels"])

        """
        assert mode in ["train", "test", "val"]

        if mode == "train":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/train"

        elif mode == "val":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/val"

        else:
            filename = f"{destination_dir}/{dataset_name}_test.tgz"
            foldername = f"{destination_dir}/{dataset_name}_test"
            # self.root_dir = foldername

        url = DATASET[f"{dataset_name}"][f"{mode}_url"]

        if download and not os.path.isdir(foldername) is True:
            if not os.path.isfile(filename):
                download_dataset(
                    filename,
                    url=url,
                )
            extract_split_dataset(filename, destination_dir)
        else:
            assert os.path.isdir(foldername) is True, "Dataset doesn't exists, set arg download to True!"

            print(f"{dataset_name} dataset already exists")

        self.root_dir = foldername

        self.transform = transform
        classes = os.listdir(self.root_dir)  # [join(self.root_dir, x).split('/')[3] for x in listdir(self.root_dir)]
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.imagefilename = []
        self.labels = []
        self.channels = channels

        for i in classes:
            for x in os.listdir(os.path.join(self.root_dir, i)):
                self.imagefilename.append(os.path.join(self.root_dir, i, x))
                self.labels.append(self.class_to_idx[i])

    def __getitem__(self, index):
        image, label = self.imagefilename[index], self.labels[index]

        image = np.load(image, allow_pickle=True)
        if label == 0:
            image = image[0]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.expand_dims(image, axis=2)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            image = image.float().clone().detach()  # .requires_grad_(True)  # torch.tensor(image, dtype=torch.float32)
        return image, label

    def __len__(self):
        return len(self.labels)


def visualize_samples(dataset, labels_map, fig_height=15, fig_width=15, num_cols=5, cols_rows=5) -> None:
    """Visualize samples from dataset

    Args:
        dataset (torch.utils.data.Dataset): dataset to visualize
        labels_map (dict): dict for mapping labels to number e.g `{0: "axion"}`
        fig_height (int, optional): height of visualized sample. Defaults to 15.
        fig_width (int, optional): width of visualized sample. Defaults to 15.
        num_cols (int, optional): # of columns of images in a window. Defaults to 5.
        cols_rows (int, optional): # of rows of images in a window. Defaults to 5.
    """
    # labels_map = {0: "axion", 1: "cdm", 2: "no_sub"}
    figure = plt.figure(figsize=(fig_height, fig_width))
    cols, rows = num_cols, cols_rows
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"{labels_map[label]}")
        plt.axis("off")
        # im = ToPILImage()(img)
        img = img.squeeze()
        plt.imshow(img, cmap="gray")
        # plt.imshow(img)
    plt.show()


class DeepLenseDataset_dataeff(Dataset):
    # TODO: add val-loader + splitting
    def __init__(
        self,
        destination_dir,
        mode,
        dataset_name,
        transform=None,
        download="False",
        channels=1,
    ):
        assert mode in ["train", "test", "val"]

        if mode == "train":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/train"

        elif mode == "val":
            filename = f"{destination_dir}/{dataset_name}.tgz"
            foldername = f"{destination_dir}/{dataset_name}"
            # self.root_dir = foldername + "/val"

        else:
            filename = f"{destination_dir}/{dataset_name}_test.tgz"
            foldername = f"{destination_dir}/{dataset_name}_test"
            # self.root_dir = foldername

        url = DATASET[f"{dataset_name}"][f"{mode}_url"]

        if download and not os.path.isdir(foldername) is True:
            if not os.path.isfile(filename):
                download_dataset(
                    filename,
                    url=url,
                )
            extract_split_dataset(filename, destination_dir)
        else:
            assert os.path.isdir(foldername) is True, "Dataset doesn't exists, set arg download to True!"

            print("Dataset already exists")

        self.root_dir = foldername

        self.transform = transform
        classes = os.listdir(self.root_dir)  # [join(self.root_dir, x).split('/')[3] for x in listdir(self.root_dir)]
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # self.imagefilename = []
        self.labels = []
        self.channels = channels

        # TODO: make dynamic
        if mode == "train":
            num = 87525
        elif mode == "test":
            num = 15000

        # if not os.path.exists(f"images_mmep_{mode}.npy"):

        self.images_mmep = np.memmap(
            f"images_mmep_{mode}.npy",
            dtype="int16",
            mode="w+",
            shape=(num, 150, 150),
        )

        self.labels_mmep = np.memmap(f"labels_mmep_{mode}.npy", dtype="float64", mode="w+", shape=(num, 1))

        w_index = 0
        for i in classes:
            for x in os.listdir(os.path.join(self.root_dir, i)):
                self.imagefilename = os.path.join(self.root_dir, i, x)
                image = np.load(self.imagefilename, allow_pickle=True)
                label = self.class_to_idx[i]
                if label == 0:
                    image = image[0]
                self.images_mmep[w_index, :] = image
                self.labels_mmep[w_index] = label
                self.labels.append(self.class_to_idx[i])
                w_index += 1

    def __getitem__(self, index):
        image = np.asarray(self.images_mmep[index])
        label = np.asarray(self.labels_mmep[index], dtype="int64")[0]

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        if self.channels == 3:
            image = Image.fromarray(image.astype("uint8")).convert("RGB")
        else:
            image = Image.fromarray(image.astype("uint8"))  # .convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


class CustomDataset(Dataset):
    """Create custom dataset for the given data"""

    def __init__(self, root_dir, mode, transform=None):
        assert mode in ["train", "test", "val"]

        self.root_dir = root_dir

        if mode == "train":
            self.root_dir = self.root_dir + "/train"
        elif mode == "test":
            self.root_dir = self.root_dir + "/test"
        else:
            self.root_dir = self.root_dir + "/val"

        self.transform = transform
        classes = os.listdir(self.root_dir)
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.imagefilename = []
        self.labels = []

        for i in classes:
            for x in os.listdir(os.path.join(self.root_dir, i)):
                self.imagefilename.append(os.path.join(self.root_dir, i, x))
                self.labels.append(self.class_to_idx[i])

    def __getitem__(self, index):
        image, label = self.imagefilename[index], self.labels[index]

        image = Image.open(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


class DefaultDatasetSetup:
    def __init__(self) -> None:
        # parent directory
        current_file = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file)

        self.data_dir = os.path.join(parent_directory, "../data/lenses")
        self.data_dir_temp = os.path.join(parent_directory, "../data/lenses_temp")
        self.zip_data_file = os.path.join(parent_directory, "../data/lenses.tgz")
        self.default_transform = DefaultTransformations()
        self.train_transform = self.default_transform.get_train_transform_eqv()
        self.test_transform = self.default_transform.get_test_transform()

    def get_default_cfg(self, dataset_name="Model_III"):
        self.default_dataset_cfg = {}
        self.default_dataset_cfg["dataset_name"] = dataset_name
        self.default_dataset_cfg["dataset_dir"] = "data"
        self.default_dataset_cfg["dataset"] = DATASET[self.default_dataset_cfg["dataset_name"]]
        self.default_dataset_cfg["classes"] = self.default_dataset_cfg["dataset"]["classes"]
        self.default_dataset_cfg["train_url"] = self.default_dataset_cfg["dataset"]["train_url"]

        make_directories([self.default_dataset_cfg["dataset_dir"]])

    def download_dataset(self):
        """Check if the compressed data file from gdrive exist else download in the directory"""

        if not os.path.isfile(self.zip_data_file):
            url = self.default_dataset_cfg["train_url"]
            output = self.zip_data_file
            gdown.download(url, output, quiet=False)
        else:
            print("File exists")

        if os.path.isdir(self.data_dir):
            print("Extracted folder exists")
        else:
            print("Extracting folder")
            os.system(f"tar xf {self.zip_data_file} --directory data ; mv {self.data_dir} {self.data_dir_temp}")
            splitfolders.ratio(self.data_dir_temp, output=self.data_dir, seed=1337, ratio=(0.9, 0.1))
            os.system(f"rm -r {self.data_dir_temp}")

    def setup(self):
        self.get_default_cfg()
        self.download_dataset()

    def get_default_trainset(self):
        self.trainset = CustomDataset(self.data_dir, "train", transform=self.train_transform)
        # get the number of samples in train and test set
        print(f"Train Data: {len(self.trainset)}")
        return self.trainset

    def get_default_testset(self):
        self.testset = CustomDataset(self.data_dir, "val", transform=self.test_transform)
        print(f"Test train Data: {len(self.testset)}")

        return self.testset

    def visualize_trainset(self):
        visualize_samples(self.trainset, labels_map=self.default_dataset_cfg["classes"])

    def visualize_testset(self):
        visualize_samples(self.testset, labels_map=self.default_dataset_cfg["classes"])
