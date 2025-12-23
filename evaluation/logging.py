import os
import shutil
from dislib import defaults


class Args:
    def __init__(self):
        self.dataset = "dsprites"
        self.seed = defaults.SEED
        self.batch_size = int(2**12)
        self.model = "cnn"
        self.lr = 1e-3
        self.num_epochs = 100
        self.log_dir = defaults.SAVE_PATH
        self.debug = False
        self.probe = False

    def __str__(self):
        return "Args:\n" + "\n".join(
            f"{attr}: {value}"
            for attr, value in self.__dict__.items()
            if not attr.startswith("__")
        )


def get_checkpoints(args):
    # Ensure the log directory exists and is empty
    if not os.path.exists(args.log_dir):
        raise Exception(f"Directory {args.log_dir} does not exist.")
    # os.makedirs(args.log_dir)  # Recreate the directory
    log_file = os.path.join(args.log_dir, "log.txt")
    with open(log_file, "a") as file:
        print(args, file=file)
    return log_file


def setup_logging(args):
    # Ensure the log directory exists and is empty
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)  # Remove the directory and all its contents
    os.makedirs(args.log_dir)  # Recreate the directory
    log_file = os.path.join(args.log_dir, "log.txt")
    with open(log_file, "a") as file:
        print(args, file=file)
    return log_file
