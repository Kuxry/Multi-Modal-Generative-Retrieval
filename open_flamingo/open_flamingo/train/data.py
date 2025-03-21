"""
Preprocess and load datasets for training.
"""

import functools
import io
import json
import math
import re
import random
import numpy as np
import torch
import torchvision
import webdataset as wds
from PIL import Image
import base64

from data_utils import *

Image.MAX_IMAGE_PIXELS = 1000000000
N_CHANNELS = 3
MIN_KB = 10
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image


def filter_no_caption_or_no_image(sample):
    """
    Filter out LAION samples with no caption or no image.
    """
    return ("txt" in sample) and (
        "png" in sample or "jpg" in sample or "jpeg" in sample
    )


def preprocess_laion_text(sample, tokenizer, max_tokens=32):
    """
    Preprocess text for LAION.
    Captions are truncated to 32 tokens by default.
    """
    tokenizer.padding_side = "right"
    sample = [
        (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
    ]
    text = tokenizer(
        sample,
        max_length=max_tokens,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


def preprocess_gpt_interleaved(
    info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens=256
):
    """
    Preprocess a ChatGPT-generated image-text sequence.
    """
    text = info["example"]
    text = re.sub(r"_!_IMAGE\d+_!_", "<|endofchunk|><image>", text)

    # convert images from base64 to PIL
    images = []
    for image_key in range(1, len(info["image_map"]) + 1):
        image_base64 = info["image_map"][f"_!_IMAGE{image_key}_!_"]["base64_image"]
        rawbytes = base64.b64decode(image_base64)
        images.append(Image.open(io.BytesIO(rawbytes)).convert("RGB"))

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (max_num_images - len(images_tensors), 3, 224, 224), dtype=torch.float
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )

    indices = [m.start() for m in re.finditer("<image>", text)]
    if len(indices) > max_num_images:
        start_index = indices[max_num_images - 1]
        text = text[:start_index]

    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images after truncation
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")

    return (images_tensors, (text_tensor["input_ids"], text_tensor["attention_mask"]))


def preprocess_interleaved(
    sample,
    tokenizer,
    clip_processor,
    sim_threshold,
    min_num_images,
    max_num_images,
    max_tokens=20,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """

    info = json.loads(sample[0])
    if "is_gpt" in info:
        return preprocess_gpt_interleaved(
            info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens
        )

    sentences = info["text_list"]
    sim_matrix = info["similarity_matrix"]

    # convert images from base64 to PIL and filter based on image-text similarity
    images, sentence_ixs = [], []
    for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
        if "image_base64" not in sample_image:
            continue
        image_base64 = sample_image["image_base64"]
        rawbytes = base64.b64decode(image_base64)

        sim_ix = np.argmax(sim_vec)
        sim_score = sim_vec[sim_ix]

        # filter to images >= 10KB
        # if len(rawbytes) // 1000 <= MIN_KB:
        #     continue
        if sim_score < sim_threshold:
            continue
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        images.append(image)
        sentence_ixs.append(sim_ix)
    if len(images) == 0:
        raise ValueError("No images in sample")

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (
                max_num_images - len(images_tensors),
                N_CHANNELS,
                images_tensors[0].shape[1],
                images_tensors[0].shape[2],
            ),
            dtype=torch.float,
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    # add in <image> and <eoc> tokens
    for ix in sentence_ixs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"
    text = " ".join(sentences)


    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"


    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")
    # elif (
    #     num_images == 1 and random.random() <= 0.5
    # ):  # 50% chance of keeping single image samples
    #     raise ValueError("Only one image in sample")            #############yongqi remove

    # avoid the situation where there's one <image> token and it's at the end
    if (
        num_images == 1
        and text_tensor["input_ids"][:, -1]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    ):
        raise ValueError(
            "Only one image at the end of sample, so labels will all be -100"
        )

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )

def preprocess_interleaved_i2id(
    sample,
    tokenizer,
    clip_processor,
    sim_threshold,
    min_num_images,
    max_num_images,
    max_tokens=20,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """

    info = json.loads(sample[0])
    if "is_gpt" in info:
        return preprocess_gpt_interleaved(
            info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens
        )

    sentences = info["text_list"]
    sim_matrix = info["similarity_matrix"]

    # convert images from base64 to PIL and filter based on image-text similarity
    images, sentence_ixs = [], []
    for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
        if "image_base64" not in sample_image:
            continue
        image_base64 = sample_image["image_base64"]
        rawbytes = base64.b64decode(image_base64)

        sim_ix = np.argmax(sim_vec)
        sim_score = sim_vec[sim_ix]

        # filter to images >= 10KB
        # if len(rawbytes) // 1000 <= MIN_KB:
        #     continue
        if sim_score < sim_threshold:
            continue
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        images.append(image)
        sentence_ixs.append(sim_ix)
    if len(images) == 0:
        raise ValueError("No images in sample")

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (
                max_num_images - len(images_tensors),
                N_CHANNELS,
                images_tensors[0].shape[1],
                images_tensors[0].shape[2],
            ),
            dtype=torch.float,
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    # add in <image> and <eoc> tokens
    for ix in sentence_ixs:
        sentences[ix] = f"<|endofchunk|>image numeric id<image>{sentences[ix]}"
    text = " ".join(sentences)


    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"


    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")
    # elif (
    #     num_images == 1 and random.random() <= 0.5
    # ):  # 50% chance of keeping single image samples
    #     raise ValueError("Only one image in sample")            #############yongqi remove

    # avoid the situation where there's one <image> token and it's at the end
    if (
        num_images == 1
        and text_tensor["input_ids"][:, -1]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    ):
        raise ValueError(
            "Only one image at the end of sample, so labels will all be -100"
        )

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )






def preprocess_interleaved_t2id(
    sample,
    tokenizer,
    clip_processor,
    sim_threshold,
    min_num_images,
    max_num_images,
    max_tokens=40,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """

    info = json.loads(sample[0])
    if "is_gpt" in info:
        return preprocess_gpt_interleaved(
            info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens
        )

    sentences = info["text_list"]
    sim_matrix = info["similarity_matrix"]


    # convert images from base64 to PIL and filter based on image-text similarity
    images, sentence_ixs = [], []
    for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
        if "image_base64" not in sample_image:
            continue
        image_base64 = sample_image["image_base64"]
        rawbytes = base64.b64decode(image_base64)

        sim_ix = np.argmax(sim_vec)
        sim_score = sim_vec[sim_ix]

        # filter to images >= 10KB
        # if len(rawbytes) // 1000 <= MIN_KB:
        #     continue
        if sim_score < sim_threshold:
            continue
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        images.append(image)
        sentence_ixs.append(sim_ix)
    if len(images) == 0:
        raise ValueError("No images in sample")

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (
                max_num_images - len(images_tensors),
                N_CHANNELS,
                images_tensors[0].shape[1],
                images_tensors[0].shape[2],
            ),
            dtype=torch.float,
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    # add in <image> and <eoc> tokens
    for ix in sentence_ixs:
        sentences[ix] = f"<|endofchunk|>caption:{' '.join(sentences[ix].split('#')[0].split(' ')[:25])}<image>{sentences[ix].split('#')[1]}"
    text = " ".join(sentences)


    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    # print(text)
    # print(tokenizer.encode(text))

    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        print(text)
        print(num_images)
        raise ValueError(f"Fewer than {min_num_images} images in sample")
    # elif (
    #     num_images == 1 and random.random() <= 0.5
    # ):  # 50% chance of keeping single image samples
    #     raise ValueError("Only one image in sample")            #############yongqi remove

    # avoid the situation where there's one <image> token and it's at the end
    if (
        num_images == 1
        and text_tensor["input_ids"][:, -1]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    ):
        print(text)
        print(num_images)
        raise ValueError(
            "Only one image at the end of sample, so labels will all be -100"
        )

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )






def get_mmc4_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for MMC4 / ChatGPT sequences
    """
    input_shards = args.mmc4_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_mmc4
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        # _SHARD_SHUFFLE_SIZE = 2000
        # _SHARD_SHUFFLE_INITIAL = 500
        # _SAMPLE_SHUFFLE_SIZE = 1000
        # _SAMPLE_SHUFFLE_INITIAL = 1000
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        # _SHARD_SHUFFLE_SIZE = 1
        # _SHARD_SHUFFLE_INITIAL = 1
        # _SAMPLE_SHUFFLE_SIZE = 1000
        # _SAMPLE_SHUFFLE_INITIAL = 1000
        pipeline = [wds.SimpleShardList(input_shards)]

    if "flicker30k_i2id" in args.mmc4_shards:
        preprocess_fn = functools.partial(
            preprocess_interleaved_i2id,
            clip_processor=image_processor,
            tokenizer=tokenizer,
            sim_threshold=args.mmc4_textsim_threshold,
            min_num_images=args.mmc4_min_num_images,
            max_num_images=args.mmc4_max_num_images,
        )
    else:
        preprocess_fn = functools.partial(
            preprocess_interleaved,
            clip_processor=image_processor,
            tokenizer=tokenizer,
            sim_threshold=args.mmc4_textsim_threshold,
            min_num_images=args.mmc4_min_num_images,
            max_num_images=args.mmc4_max_num_images,
        )

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                )
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(args.batch_size_mmc4, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_mmc4 * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def get_text2id_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for MMC4 / ChatGPT sequences
    """
    input_shards = args.laion_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_laion
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        # _SHARD_SHUFFLE_SIZE = 2000
        # _SHARD_SHUFFLE_INITIAL = 500
        # _SAMPLE_SHUFFLE_SIZE = 1000
        # _SAMPLE_SHUFFLE_INITIAL = 1000
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        # _SHARD_SHUFFLE_SIZE = 1
        # _SHARD_SHUFFLE_INITIAL = 1
        # _SAMPLE_SHUFFLE_SIZE = 1000
        # _SAMPLE_SHUFFLE_INITIAL = 1000
        pipeline = [wds.SimpleShardList(input_shards)]


    preprocess_fn = functools.partial(
        preprocess_interleaved_t2id,
        clip_processor=image_processor,
        tokenizer=tokenizer,
        sim_threshold=args.mmc4_textsim_threshold,
        min_num_images=args.mmc4_min_num_images,
        max_num_images=args.mmc4_max_num_images,
    )

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                )
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(args.batch_size_laion, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_laion * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)
def get_laion_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for LAION data
    """
    input_shards = args.laion_shards


    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)

    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_laion

        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]

    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # create two preprocess functions that take in the passed in image_processor and tokenizer
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_laion_text, tokenizer=tokenizer)

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.to_tuple("jpg;png;jpeg", "txt", handler=log_and_continue),
            wds.batched(args.batch_size_laion, partial=False),
            wds.map_tuple(
                preprocess_image_fn, preprocess_text_fn, handler=log_and_continue
            ),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_laion * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples


    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_fn(dataset_type):
    """
    Helper function to get the dataset function based on the dataset type
    """
    if dataset_type == "image_text":
        return get_laion_dataset
    elif dataset_type == "mmc4":
        return get_mmc4_dataset
    elif dataset_type == "text2id":
        return get_text2id_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor, tokenizer, dataset_type, epoch=0):
    """
    Interface for getting the webdatasets
    """
    return get_dataset_fn(dataset_type)(
        args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer
    )
