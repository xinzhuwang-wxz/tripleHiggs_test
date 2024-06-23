import torch
from torch.utils.data import DataLoader
from collate import collate_fn
from init_dataset import init_dataset

def retrieve_dataloaders(batch_size, num_workers=0, datadir='./data'):
    """
    Initialize dataloaders for MPS.

    Parameters:
    - batch_size: int, the size of the batches
    - num_workers: int, number of worker processes for data loading
    - datadir: str, directory containing the data

    Returns:
    - dataloaders: dict, dataloaders for train, test, and valid datasets
    """

    # Initialize datasets
    datasets = init_dataset(datadir, shuffle=True)

    # Define a collate function with additional arguments
    collate = lambda data: collate_fn(data, scale=1, add_beams=True, beam_mass=1)

    # Construct PyTorch dataloaders for each dataset split
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size, 
                                     shuffle=(split == 'train'),  # Shuffle only for training dataset
                                     pin_memory=True,
                                     drop_last=True if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate)
                   for split, dataset in datasets.items()}

    return dataloaders

if __name__ == "__main__":
    batch_size = 32
    num_workers = 0
    datadir = "data/raw_data"

    # Retrieve dataloaders
    dataloaders = retrieve_dataloaders(batch_size, num_workers, datadir=datadir)

    print("Keys in dataloaders:", dataloaders.keys())
