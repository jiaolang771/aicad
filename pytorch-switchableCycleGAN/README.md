# pytorch-abcd-switchableCycleGAN
# Image-to-Image Translation Training Script


This script is adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md

## Description

This script provides functionality for training image-to-image translation models. The framework supports multiple models such as `pix2pix`, `cyclegan`, and `colorization` along with various dataset modes.

## Key Features

- Training support for several models: `pix2pix`, `cyclegan`, and more.
- Different dataset modes like `aligned`, `unaligned`, `single`, and `colorization`.
- Resume and continue training capability with `--continue_train`.
- Periodic visualization and saving of images, plotting of losses, and model checkpoints.
- Options to adjust the training flow as per user requirements.

## Dependencies

- PyTorch
- Visdom (for visualizations)

## How to Run

1. Setup your dataset. Ensure it's in the right format depending on the model and dataset mode you are using.
2. Execute the training script with the necessary command-line arguments. For example:
   
    For a CycleGAN model:
    ```bash
    python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    ```

    For a pix2pix model:
    ```bash
    python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
    ```

3. Monitor the training progress in the console. Images, plots, and model checkpoints will be saved periodically as per the provided options.

## Options

Many training options can be adjusted via command-line arguments. Refer to `options/base_options.py` and `options/train_options.py` for a complete list and descriptions of available options.

## Useful Links

- [Training and Test Tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md)
- [Frequently Asked Questions](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md)

## License

Please adhere to the licensing terms if you utilize or adapt this code for your requirements.

