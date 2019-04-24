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


