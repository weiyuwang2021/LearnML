import torch
import numpy as np
from .CFG import CFG
from .utils import get_transforms

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.dataset = dataset
        self.captions = self.dataset['en_text']
        self.images = self.dataset['image']
        self.encoded_captions = tokenizer(
            self.dataset['en_text'], padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = self.transforms(image=np.array(self.images[idx]))['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)
    
def build_loaders(dataset, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataset[mode],
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader