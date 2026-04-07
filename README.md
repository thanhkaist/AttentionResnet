# AttentionResnet
Classification with backbone Resnet and attentions: SE-Channel Attention, BAM - (Spatial Attention, Channel Attention, Joint Attention), CBAM - (Spatial Attention, Channel Attention, Joint Attention)

Dataset: CIFA100

To run the experiments:

```
$./run.sh
```

To test model

```
$./run_test.sh
```

## Training with a custom dataset

You can train any model on your own image dataset using the `--dataset custom` flag.

### 1. Prepare your data

Organise your images in the standard `ImageFolder` structure:

```
/path/to/my_dataset/
    train/
        cat/
            img001.jpg
            img002.jpg
            ...
        dog/
            img001.jpg
            ...
    val/
        cat/
            img101.jpg
            ...
        dog/
            img101.jpg
            ...
```

Each sub-folder name becomes a class label; the model will output one logit per sub-folder found under `train/`.

### 2. Run training

```bash
python main.py \
    --dataset custom \
    --data_dir /path/to/my_dataset \
    --num_classes <number_of_classes> \
    --image_size 500 \
    --model cbam_resnet50 \
    --attention joint \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.01
```

Key arguments:

| Argument | Description | Default |
|---|---|---|
| `--dataset` | `cifar100` or `custom` | `cifar100` |
| `--data_dir` | Root directory that contains `train/` and `val/` sub-folders | `./data` |
| `--num_classes` | Number of classes in your dataset | `100` |
| `--image_size` | Images are resized to this square size before training | `32` |

> **Tip for ~500×500 images:** pass `--image_size 500` (or any value that suits your dataset).
> A random crop of that size is taken during training and a centre crop during validation.
> The model uses `AdaptiveAvgPool2d` internally, so **any** image size is supported.

