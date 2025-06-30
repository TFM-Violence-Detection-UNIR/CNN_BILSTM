
import random
import os
import shutil

def split_data(root_dir, train_pct=0.8, seed=None):
    """
    Splits the contents of each class-folder under `root_dir` into train and test sets.

    Args:
        root_dir (str): Path to your dataset folder (e.g. "datasets/hockey_fights").
                        Inside this folder you should have one folder per class
                        (e.g. "fight", "no_fight").
        train_pct (float): Fraction of data to allocate to training (0 < train_pct < 1).
                           The remainder (1 - train_pct) goes to testing.
        seed (int, optional): Random seed for reproducibility. If None, shuffle is random.

    After running, you’ll end up with:
        root_dir/
            fight/        # original data untouched
            no_fight/     # original data untouched
            train/
                fight/
                no_fight/
            test/
                fight/
                no_fight/
    """
    if not 0 < train_pct < 1:
        raise ValueError("train_pct must be between 0 and 1")

    if seed is not None:
        random.seed(seed)

    # Discover class subfolders (ignore any existing train/test directories)
    classes = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d not in ("train", "test")
    ]

    for cls in classes:
        src_dir   = os.path.join(root_dir, cls)
        train_dir = os.path.join(root_dir, "train", cls)
        test_dir  = os.path.join(root_dir, "test", cls)

        # Create train/test dirs if needed
        for d in (train_dir, test_dir):
            if not os.path.exists(d):
                os.makedirs(d)
                print(f"Created directory {d}")
            else:
                print(f"Directory {d} already exists; skipping creation.")

        # List and shuffle files
        files = [
            f for f in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, f))
        ]
        random.shuffle(files)

        split_idx = int(len(files) * train_pct)
        train_files = files[:split_idx]
        test_files  = files[split_idx:]

        # Copy files
        for fname in train_files:
            shutil.copy2(os.path.join(src_dir, fname),
                         os.path.join(train_dir, fname))
        for fname in test_files:
            shutil.copy2(os.path.join(src_dir, fname),
                         os.path.join(test_dir, fname))

        print(f"{cls}: {len(train_files)} → train, {len(test_files)} → test")
        
        
split_data("datasets/RLVS", train_pct=0.8, seed=42)