import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class DocumentDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initializes the DocumentDataset with the directory containing the images and an optional transform.

        Args:
            root_dir (str): The root directory containing class subdirectories with images.
            transform (callable, optional): A function/transform to apply to the images.

        Attributes:
            root_dir (str): The root directory containing the images.
            transform (callable, optional): A transform function to apply to the images.
            classes (list): List of class names found in the root directory.
            image_paths (list): List of paths to all images in the dataset.
            labels (list): List of labels corresponding to the class of each image.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Retrieves the image and its corresponding label at the specified index.

        Args:
            index (int): The index of the image and label to retrieve.

        Returns:
            tuple: A tuple containing the image and its label. The image is loaded
            and converted to RGB format, and the transform is applied if specified.
        """

        img_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, label
