import argparse

"""
This is the configs for our model.
You can set all hyperparameters in here.
"""

def argparsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default='results/', help='path of result')
    parser.add_argument('--load', action='store_true', help='load from checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay hyperparameter')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 70,80,90,95],help='Decrease learning rate at these epochs.')
    parser.add_argument('--checkpoint', type=str, default='checkpoint', help='checkpoint of the detector')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--attention', type=str, default='no')
    parser.add_argument('--norm', type=str, default='bn')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='Dataset to use: "cifar100" (default) or "custom"')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the dataset directory. For custom datasets, should follow '
                             'ImageFolder structure: <data_dir>/train/<class>/ and '
                             '<data_dir>/val/<class>/')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='Number of output classes (default: 100 for CIFAR-100)')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Input image size after resizing (default: 32 for CIFAR-100). '
                             'For custom datasets with larger images, e.g. 224 or 500.')

    return parser.parse_args()

args = argparsing()

class Configs:
    weight_decay = args.weight_decay
    results_path = args.results_path
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.learning_rate
    test  =  args.test
    model = args.model
    attention = args.attention
    norm = args.norm
    schedule = args.schedule
    gpu = True
    dataset = args.dataset
    data_dir = args.data_dir
    num_classes = args.num_classes
    image_size = args.image_size


