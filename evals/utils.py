from datasets import load_dataset, load_from_disk


def load_dataset_my_way(
    dataset_name, name=None, split="train", processing_function=None
):
    try:
        dataset = load_dataset(dataset_name, name=name)
        if split:
            dataset = dataset[split]

        return dataset
    except Exception as e:
        dataset = load_from_disk(dataset_name)
        if split:
            dataset = dataset[split]
        return dataset
