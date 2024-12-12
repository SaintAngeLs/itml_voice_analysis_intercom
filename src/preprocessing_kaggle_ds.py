import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.stride_tricks as stride_tricks
import librosa
import multiprocessing as mp



def overlapping_frames(x: np.ndarray, frame_size: int, overlap_factor: float) -> np.ndarray:
    assert len(x) >= frame_size, f"Frame size ({frame_size}) exceeds length of x ({len(x)})."
    assert overlap_factor >= 0.0 and overlap_factor < 1.0,\
        f"Overlap factor ({overlap_factor}) has to be from range [0.0, 1.0)."

    n = x.shape[0]
    stride_len = x.strides[0]

    stride_size = int(frame_size - np.floor(frame_size * overlap_factor))
    window_count = int(np.floor((n - frame_size)/stride_size)) + 1

    frames = stride_tricks.as_strided(x, shape=(window_count, frame_size),
                                      strides=(stride_len*stride_size, stride_len))

    return frames



def overlapping_frames2D(X: np.ndarray, frame_size: int, overlap_factor: float) -> np.ndarray:
    assert len(X.shape) == 2, "X is not a 2D matrix."
    height, width = X.shape
    assert width >= frame_size, f"Frame size ({frame_size}) exceeds length of x ({width})."

    assert overlap_factor >= 0.0 and overlap_factor < 1.0,\
        f"Overlap factor ({overlap_factor}) has to be from range [0.0, 1.0)."

    stride_size = int(frame_size - np.floor(frame_size * overlap_factor))
    # Remove offset from the beginning of the data, its just background noise
    offset = (width - frame_size) % stride_size
    X = X[:, offset:]
    height, width = X.shape
    window_count = (width - frame_size) // stride_size + 1

    stride_height, stride_width = X.strides
    frames = stride_tricks.as_strided(X, shape=(window_count, height, frame_size),
                                      strides=(stride_width*stride_size, stride_height, stride_width))

    return frames



def process_file(
        input_path: str,
        output_path: str,
        augmentations: dict[str]
):
    sr = None
    if "sr" in augmentations:
        sr = float(augmentations["sr"])

    n_mels = 256
    if "target_height" in augmentations:
        n_mels = int(augmentations["target_height"])

    y, sr = librosa.load(input_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logS = librosa.power_to_db(S)

    plt.imsave(output_path, logS, cmap="bwr")




def preprocess_dir(input_path: str, output_path: str, dir: str, augmentations: dict[str],
                   test_set: set[str], validation_set: set[str]):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    print(f"[LOG] Processing directory '{dir}'")
    full_input_path = os.path.join(input_path, dir)
    for entry in os.scandir(full_input_path):
        if entry.is_dir():
            print("[LOG] \tEntry is a directory, skipping")
            continue

        input_file_path = os.path.join(dir, entry.name)
        if input_file_path in test_set:
            output_set = "test"
        elif input_file_path in validation_set:
            output_set = "validation"
        else:
            output_set = "train"

        output_dir_path = os.path.join(output_path, output_set, dir)
        os.makedirs(output_dir_path, exist_ok=True)

        filename = input_file_path.removesuffix(".wav") + ".png"
        full_output_filename = os.path.join(output_path, output_set, filename)
        process_file(entry.path, full_output_filename, augmentations)

    print(f"[LOG] Directory '{dir}' has been processed")


def main():
    import argparse
    import time
    import json

    # TODO: Create some usage
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    data_path = config["data_path"]
    ds_path = os.path.join(data_path, "tensorflow-speech-recognition-challenge", "train")
    audio_path = os.path.join(ds_path, "audio")
    output_path = os.path.join(data_path, config["output_dir"])
    dirs = config["dirs"]
    augmentations = config["augmentations"]

    test_set = set([s.strip() for s in open(os.path.join(ds_path, "testing_list.txt")).readlines()])
    validation_set = set([s.strip() for s in open(os.path.join(ds_path, "validation_list.txt")).readlines()])

    pool = mp.Pool(4)
    print("START PROCESSING")
    start = time.time()
    results = [pool.apply_async(preprocess_dir,
                                [audio_path, output_path, dir, augmentations, test_set, validation_set])
               for dir in dirs]

    for r in results:
        r.get()

    end = time.time()
    print("END PROCESSING")
    print(f"Elapsed time: {end - start} s")


if __name__ == "__main__":
    main()