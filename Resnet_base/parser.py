import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test", help="run name")
    parser.add_argument("--save_id", type=str, help="model ID")
    parser.add_argument("--load_id", type=str, default=None, help="model ID to load")
    parser.add_argument("--resume_from", type=int, default=-1, help="epoch to load")

    parser.add_argument("--task", type=str, help="Task: train/test")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda device to use")
    parser.add_argument("--model", type=str, default="nodown", help="architecture name")
    parser.add_argument("--freeze", action='store_true', help="Freeze all layers except the last one")

    parser.add_argument("--num_epochs", type=int, default=100, help="# of epoches at starting learning rate")

    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--lr_decay_epochs",type=int, default=-1, help="Number of epochs without loss reduction before lowering the learning rate by 10x")

    parser.add_argument("--save_weights", action='store_true', help="Save weights during training")
    parser.add_argument("--save_scores", action='store_true', help="")
    parser.add_argument("--features", action='store_true', help="extract features before linear layer")

    parser.add_argument("--split_path", type=str, help="Path to split files")
    parser.add_argument("--data_root", type=str, help="Path to dataset")
    parser.add_argument("--data", type=str, help="Dataset specifications")

    parser.add_argument("--batch_size", type=int, default=64, help='Dataloader batch size')
    parser.add_argument("--num_threads", type=int, default=-1, help='# threads for loading data')

    # POISONING PARAMETER
    parser.add_argument("--poison_rate", type=float, default=0.0, help='Percentage of mislabeled samples (0.0-1.0). E.g., 0.1 = 10%% poisoning')

    # data-augmentation 
    parser.add_argument("--resize_prob", type=float, default=0.0)
    parser.add_argument("--resize_scale", type=float, nargs='+', default=1.0)
    parser.add_argument("--resize_ratio", type=float, nargs='+', default=1.0)
    parser.add_argument("--resize_size", type=int, default=256)

    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--jpeg_prob", type=float, default=0.0)
    parser.add_argument("--jpeg_qual", type=int, nargs='+', default=75)

    parser.add_argument("--blur_prob", type=float, default=0.0)
    parser.add_argument("--blur_sigma", type=float, nargs='+', default=0.5)

    parser.add_argument("--patch_size", type=int, default=-1)
    
    return parser