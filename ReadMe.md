# Video Classification Beginner Project

This project is a complete beginner-friendly deep learning pipeline for simple video classification using PyTorch, OpenCV, and torchvision.

The project teaches the full workflow:

1. download or find videos
2. load videos
3. inspect video properties
4. extract frames
5. label frame sequences
6. create a sequence dataset
7. resize frames
8. normalize frames
9. batching
10. DataLoader
11. train model
12. evaluate
13. save model
14. load model for inference

The default classes are:

- `sitting`
- `standing`

The preferred public download is attempted first. If downloading fails, the project automatically creates a tiny synthetic motion dataset using OpenCV so the whole pipeline still works on CPU.

## Project Structure

```text
video_dl_project/
│
├── data/
│   ├── raw_videos/
│   ├── extracted_frames/
│   ├── sequences/
│   ├── train/
│   ├── val/
│   └── test/
│
├── outputs/
│   ├── models/
│   ├── plots/
│   └── predictions/
│
├── src/
│   ├── download_videos.py
│   ├── inspect_videos.py
│   ├── extract_frames.py
│   ├── prepare_sequences.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

## What This Project Does

- downloads or creates a tiny 2-class video dataset
- reads videos with OpenCV
- inspects total frames, FPS, width, and height
- extracts all frames into folders
- samples fixed-length frame sequences
- builds a custom PyTorch video dataset
- trains a CPU-friendly video classification model
- evaluates on a test set
- saves the best model and plots
- predicts on one new video

## How Video Download Works

`src/download_videos.py` first tries a public download URL for each class.

If internet access is unavailable or the download fails, the script creates short synthetic videos:

- `sitting` videos have slower stick-figure movement
- `standing` videos have faster stick-figure movement

This fallback is helpful for teaching because the whole pipeline remains runnable.

## How Frame Extraction Works

`src/extract_frames.py`:

- opens each video with OpenCV
- reads every frame
- saves frames as `.jpg` images
- stores them inside `data/extracted_frames/<class_name>/<video_name>/`

## How Sequences Are Built

`src/prepare_sequences.py`:

- takes extracted frame folders
- samples a fixed number of frames per video
- default sequence length is `8`
- copies selected frames into sequence folders
- splits sequences into `train`, `val`, and `test`

Each sample is one folder of ordered frames.

## Input Shapes For Learning

One sample has shape:

```text
[T, C, H, W]
```

Example:

```text
[8, 3, 128, 128]
```

One batch has shape:

```text
[B, T, C, H, W]
```

Example:

```text
[2, 8, 3, 128, 128]
```

## Preprocessing

For each frame:

- resize to `128 x 128`
- convert to tensor
- normalize with mean `(0.5, 0.5, 0.5)` and std `(0.5, 0.5, 0.5)`

Training also includes light horizontal flip augmentation.

## Model

The model is intentionally simple and CPU-friendly:

1. a small CNN extracts features from each frame
2. temporal average pooling combines features across time
3. a small classifier predicts the video class

This is good for teaching because it separates:

- frame feature extraction
- temporal aggregation
- final classification

## What Each Script Does

- `src/download_videos.py`
  Downloads videos or creates a fallback video dataset.

- `src/inspect_videos.py`
  Prints total frames, FPS, width, and height for each video.

- `src/extract_frames.py`
  Extracts frames from every video.

- `src/prepare_sequences.py`
  Samples fixed-length sequences and creates train, val, and test splits.

- `src/dataset.py`
  Defines the custom `VideoSequenceDataset` and DataLoaders.

- `src/model.py`
  Defines the small video classification model.

- `src/train.py`
  Trains the model, saves the best model, class names, and training plots.

- `src/evaluate.py`
  Evaluates the model on test data and saves a confusion matrix.

- `src/predict.py`
  Loads one new video, preprocesses it, predicts the class, and prints confidence.

- `src/utils.py`
  Stores helper functions, constants, paths, plotting, and JSON helpers.

## Installation

Open a terminal inside `video_dl_project/` and run:

```bash
pip install -r requirements.txt
```

## Exact Command Order To Run The Full Project

```bash
cd video_dl_project
pip install -r requirements.txt
python src/download_videos.py
python src/inspect_videos.py
python src/extract_frames.py
python src/prepare_sequences.py
python src/train.py --epochs 8 --batch_size 2
python src/evaluate.py
python src/predict.py
```

You can also predict on a specific video:

```bash
python src/predict.py --video_path data/raw_videos/sitting/sitting_00.mp4
```

## Outputs

After running the pipeline, the project saves:

- model weights to `outputs/models/best_video_model.pth`
- class names to `outputs/models/class_names.json`
- training history to `outputs/plots/training_history.json`
- training curves to `outputs/plots/training_curves.png`
- confusion matrix to `outputs/plots/confusion_matrix.png`
- prediction preview images to `outputs/predictions/`

## CPU Notes

- the project uses CPU only
- the dataset is intentionally tiny
- sequence length is short
- the model is lightweight
- the batch size is small
- the default training setup is chosen for classroom-friendly execution
- `8` epochs is a good default for the synthetic fallback dataset

## Project Flow

The scripts connect in this order:

1. `download_videos.py` gets or creates videos
2. `inspect_videos.py` shows video properties
3. `extract_frames.py` converts videos into image frames
4. `prepare_sequences.py` builds fixed-length labeled sequences
5. `dataset.py` loads sequences into PyTorch
6. `model.py` defines the video classifier
7. `train.py` trains and saves the best model
8. `evaluate.py` checks test accuracy
9. `predict.py` loads the model and predicts one new video