import argparse
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Iterable, Tuple, Union, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from . import config as config_lib
from . import dataset_spec as dataset_spec_lib
from . import pipeline as torch_pipeline
from .pipeline import worker_init_fn_
from .utils import Split

DL = Union[DataLoader, Iterable[Tuple[Tensor, ...]]]


def get_dataspecs(args: argparse.Namespace, source: str) -> Tuple[Any, Any, Any]:
    # Recovering data
    data_config = config_lib.DataConfig(args=args)
    episod_config = config_lib.EpisodeDescriptionConfig(args=args)

    use_bilevel_ontology = False
    use_dag_ontology = False

    # Enable ontology aware sampling for Omniglot and ImageNet.
    if source == "omniglot":
        # use_bilevel_ontology_list[sources.index('omniglot')] = True
        use_bilevel_ontology = True
    if source in ["ilsvrc_2012", "ilsvrc_2012_v2"]:
        use_dag_ontology = True

    episod_config.use_bilevel_ontology = use_bilevel_ontology
    episod_config.use_dag_ontology = use_dag_ontology

    dataset_records_path: Path = data_config.path / source
    # Original codes handles paths as strings:
    dataset_spec = dataset_spec_lib.load_dataset_spec(str(dataset_records_path))

    return dataset_spec, data_config, episod_config


def get_dataloader(
    args: argparse.Namespace,
    source: str,
    batch_size: int,
    split: Split,
    version: str,
    episodic: bool,
):
    dataset_spec, data_config, episod_config = get_dataspecs(args, source)
    num_classes = len(dataset_spec.get_classes(split=split))

    pipeline_fn: Callable[..., Dataset]
    data_loader: DL
    if version == "pytorch":
        pipeline_fn = cast(
            Callable[..., Dataset],
            (
                torch_pipeline.make_episode_pipeline
                if episodic
                else torch_pipeline.make_batch_pipeline
            ),
        )
        dataset: Dataset = pipeline_fn(
            dataset_spec=dataset_spec,
            data_config=data_config,
            split=split,
            episode_descr_config=episod_config,
        )

        pprint(f"Batch size per worker {batch_size}")
        worker_init_fn = partial(worker_init_fn_, seed=args.seed)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=data_config.num_workers,
            worker_init_fn=worker_init_fn,
        )
    elif version == "tf":
        import gin
        import tensorflow as tf
        from .original_meta_dataset.data import pipeline as tf_pipeline

        tf.compat.v1.disable_eager_execution()
        tf_pipeline_fn: Callable[..., Any]
        tf_pipeline_fn = (
            tf_pipeline.make_one_source_episode_pipeline
            if episodic
            else tf_pipeline.make_one_source_batch_pipeline
        )

        GIN_FILE_PATH = (
            "src/datasets/original_meta_dataset/learn/gin/setups/data_config.gin"
        )
        gin.parse_config_file(GIN_FILE_PATH)

        tf_dataset: Any = tf_pipeline_fn(
            dataset_spec=dataset_spec,
            use_dag_ontology=episod_config.use_dag_ontology,
            use_bilevel_ontology=episod_config.use_bilevel_ontology,
            episode_descr_config=episod_config,
            split=split,
            batch_size=batch_size,
            image_size=84,
            shuffle_buffer_size=300,
        )

        iterator: Iterable = tf_dataset.make_one_shot_iterator().get_next()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        session = tf.compat.v1.Session()

        data_loader = infinite_loader(session, iterator, episodic, mean, std)
    else:
        raise ValueError(
            f"Wrong loader version, got {version}, \
                           expected to be in ['pytorch', 'tf']"
        )

    return data_loader, num_classes


def to_torch_imgs(img: np.ndarray, mean: Tensor, std: Tensor) -> Tensor:
    t_img: Tensor = torch.from_numpy(np.transpose(img, (0, 3, 1, 2)))
    t_img -= mean
    t_img /= std

    return t_img


def to_torch_labels(a: np.ndarray) -> Tensor:
    return torch.from_numpy(a).long()


def infinite_loader(
    session: Any, iterator: Iterable, episodic: bool, mean: Tensor, std: Tensor
) -> Iterable[Tuple[Tensor, ...]]:
    while True:
        (e, source_id) = session.run(iterator)
        if episodic:
            yield (
                to_torch_imgs(e[0], mean, std).unsqueeze(0),
                to_torch_imgs(e[3], mean, std).unsqueeze(0),
                to_torch_labels(e[1]).unsqueeze(0),
                to_torch_labels(e[4]).unsqueeze(0),
            )
        else:
            yield (to_torch_imgs(e[0], mean, std), to_torch_labels(e[1]))
