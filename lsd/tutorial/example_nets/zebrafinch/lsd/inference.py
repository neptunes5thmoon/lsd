import json
import logging
import os
import subprocess

import click
import daisy
import numpy as np
import tensorflow.compat.v1 as tf
from funlib.persistence import Array, open_ds, prepare_ds
from skimage.transform import rescale


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


def spawn_worker(
    setup_dir,
    checkpoint,
    input_tensor,
    output_tensor,
    num_outputs,
    channels,
    voxel_size,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    billing,
    local=True,
    min_raw=(0,),
    max_raw=(255,),
    mask_containers=list(),
    mask_datasets=list(),
    instance: bool = False,
    time_limit: int = 2000,
):
    def run_worker():
        mask_args = []
        for mask_container, mask_dataset in zip(mask_containers, mask_datasets):
            mask_args.extend(["-mc", mask_container, "-md", mask_dataset])
        process_cmd = [
            "python",
            "/groups/saalfeld/home/heinrichl/dev/fly-organelles-lsd/lsd/lsd/tutorial/example_nets/zebrafinch/lsd/inference.py",
            "start-worker",
            "-setup",
            f"{setup_dir}",
            "-ckpt",
            f"{checkpoint}",
        ]
        for it in input_tensor:
            process_cmd.extend(["-input_tensor", f"{it}"])
        for ot in output_tensor:
            process_cmd.extend(["-output_tensor", f"{ot}"])
        process_cmd.extend(["-no", f"{num_outputs}"])
        for ch in channels:
            process_cmd.extend(["-cs", f"{ch}"])
        process_cmd.extend(
            [
                "-vs",
                *map(str, voxel_size),
                "-oc",
                f"{out_container}",
                "-od",
                f"{out_dataset}",
            ]
        )
        for in_container_iter, in_dataset_iter, min_raw_iter, max_raw_iter in zip(
            in_container, in_dataset, min_raw, max_raw
        ):
            process_cmd.extend(
                [
                    "-ic",
                    f"{in_container_iter}",
                    "-id",
                    f"{in_dataset_iter}",
                    "--min-raw",
                    f"{min_raw_iter}",
                    "--max-raw",
                    f"{max_raw_iter}",
                ]
            )
        process_cmd.extend(
            [
                "--instance",
                f"{instance}",
                *mask_args,
            ]
        )
        if local:
            subprocess.run(process_cmd)
        else:
            subprocess.run(
                [
                    "bsub",
                    "-P",
                    f"{billing}",
                    "-J",
                    "pred",
                    "-q",
                    "gpu_tesla",
                    "-n",
                    "2",
                    "-gpu",
                    "num=1",
                    "-o",
                    "prediction_logs/out",
                    "-e",
                    "prediction_logs/err",
                    "-W",
                    f"{time_limit}",
                    *process_cmd,
                ]
            )

    return run_worker


def load_eval_model(setup_dir, checkpoint, device=None):
    graph = tf.Graph()
    session = tf.Session(graph=graph)

    with graph.as_default():
        meta_graph_file = os.path.join(setup_dir, "config.meta")  # f"{checkpoint}.meta"
        saver = tf.train.import_meta_graph(meta_graph_file, clear_devices=True)
        saver.restore(session, os.path.join(setup_dir, checkpoint))
        all_names = list(n.name for n in tf.get_default_graph().as_graph_def().node)
        print([n for n in all_names if "Placeholder" in n])
    return session


@cli.command()
@click.option("-setup", "--setup_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-ckpt", "--checkpoint", type=str)
@click.option("-input_tensor", type=str, multiple=True)
@click.option("-output_tensor", type=str, multiple=True)
@click.option("-no", "--num_outputs", type=int)
@click.option(
    "-cs", "--channels", type=str, multiple=True
)  # 0:ecs,1:plasma_membrane,2:ecs (comma seperated number:dataset_name)
@click.option("-vs", "--voxel_size", nargs=3, type=int)
@click.option("-oc", "--out_container", type=str)  # zarr file
@click.option("-od", "--out_dataset", type=str)  # zarr group
@click.option(
    "-ic",
    "--in_container",
    type=click.Path(exists=True, file_okay=False),
    multiple=True,
)
@click.option("-id", "--in_dataset", type=str, multiple=True)
@click.option("-w", "--workers", type=int, default=1)  # number of workers
@click.option(
    "-roi",
    "--roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)  # "[0:480000,10000:120000,15000:180000]" (in world coordinates)
@click.option("--local/--bsub", default=True)
@click.option("--billing", default=None)
@click.option("--min-raw", type=float, default=[0,], multiple=True)
@click.option("--max-raw", type=float, default=[255,], multiple=True)
@click.option(
    "-mc",
    "--mask-container",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)  # ignore
@click.option(
    "-md",
    "--mask-dataset",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)  # ignore
@click.option("--instance", type=bool, default=False)  # this is for affinities
@click.option(
    "-tw", "--per-block-time-estimate", type=float, default=30, help="in seconds"
)
def predict(
    setup_dir,
    checkpoint,
    input_tensor,
    output_tensor,
    num_outputs,
    channels,
    voxel_size,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    workers,
    roi,
    local,
    billing,
    min_raw,
    max_raw,
    mask_container,
    mask_dataset,
    instance,
    per_block_time_estimate,
):
    if not local:
        assert billing is not None
    parsed_channels = []
    for channel_split in channels:
        parsed_channels.append(
            channel.split(":") for channel in channel_split.split(",")
        )
    assert os.path.exists(os.path.join(setup_dir, checkpoint + ".meta"))
    input_ds = []
    for in_container_iter, in_dataset_iter in zip(in_container, in_dataset):
        input_ds.append(
            open_ds(os.path.join(in_container_iter, in_dataset_iter), voxel_size=voxel_size)
        )
    with open(os.path.join(setup_dir, "config.json"), "r") as f:
        net_config = json.load(f)

    if roi is not None:
        parsed_start, parsed_end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in roi.strip("[]").split(",")
            ]
        )
        parsed_roi = daisy.Roi(
            daisy.Coordinate(parsed_start),
            daisy.Coordinate(parsed_end) - daisy.Coordinate(parsed_start),
        )
    else:
        parsed_roi = input_ds[0].roi

    total_write_roi = input_ds[0].roi
    output_voxel_size = daisy.Coordinate(voxel_size)
    read_shape = daisy.Coordinate(net_config["input_shape"]) * input_ds[0].voxel_size
    write_shape = daisy.Coordinate(net_config["output_shape"]) * output_voxel_size
    context = (read_shape - write_shape) / 2
    read_roi = daisy.Roi((0,) * read_shape.dims, read_shape)
    write_roi = read_roi.grow(-context, -context)
    print(f"{context=}")
    print(f"{read_shape=}")
    print(f"{write_shape=}")
    total_write_roi = parsed_roi.snap_to_grid(input_ds[0].voxel_size)
    total_read_roi = total_write_roi.grow(context, context)
    n_blocks = np.prod((total_write_roi / write_shape).shape)
    print(f"{total_write_roi=}")
    print(f"{total_read_roi=}")
    time_per_worker = (per_block_time_estimate * n_blocks) / workers
    time_limit = int(np.ceil(time_per_worker / 60.0))
    out_container = os.path.join(setup_dir, out_container)

    if not instance:
        out_datasets = []
        for channel_iter in parsed_channels:
            out_datasets.append([])
            for indexes, channel in channel_iter:
                num_channels = (
                    1
                    if "-" not in indexes
                    else len(
                        range(int(indexes.split("-")[0]), int(indexes.split("-")[1]))
                    )
                )
                prepare_ds(
                    f"{out_container}/{out_dataset}/{channel}",
                    shape=(num_channels,)
                    + tuple((total_write_roi / output_voxel_size).shape),
                    voxel_size=output_voxel_size,
                    chunk_shape=(1,) + tuple((write_roi / output_voxel_size).shape),
                    dtype=np.uint8,
                    axis_names=["c^", "z", "y", "x"],
                    mode="w",
                )
    else:
        raise NotImplementedError()
        # num_channels = num_outputs
        # assert len(parsed_channels) == 1
        # indexes, channel = parsed_channels[0]
        # for i in range(0, num_channels, 3):
        #     prepare_ds(
        #         f"{out_container}/{out_dataset}/{channel}__{i}",
        #         total_roi=total_write_roi,
        #         voxel_size=output_voxel_size,
        #         write_size=write_roi.shape,
        #         dtype=np.float32,
        #         num_channels=min(3, num_channels - i),
        #     )

    task = daisy.Task(
        "test_server_task",
        total_roi=total_read_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=spawn_worker(
            setup_dir,
            checkpoint,
            input_tensor,
            output_tensor,
            num_outputs,
            channels,
            voxel_size,
            out_container,
            out_dataset,
            in_container,
            in_dataset,
            billing,
            local,
            min_raw,
            max_raw,
            mask_container,
            mask_dataset,
            instance,
            time_limit,
        ),
        check_function=None,
        read_write_conflict=False,
        fit="overhang",
        num_workers=workers,
        max_retries=0,
        timeout=None,
    )

    daisy.run_blockwise([task])


@cli.command()
@click.option("-setup", "--setup_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-ckpt", "--checkpoint", type=str)
@click.option("-input_tensor", type=str, multiple=True)
@click.option("-output_tensor", type=str, multiple=True)
@click.option("-no", "--num_outputs", type=int)
@click.option("-cs", "--channels", type=str, multiple=True)
@click.option("-vs", "--voxel_size", nargs=3, type=int)
@click.option("-oc", "--out_container", type=click.Path(exists=True, file_okay=False))
@click.option("-od", "--out_dataset", type=str)
@click.option("-ic", "--in_container", type=click.Path(exists=True, file_okay=False), multiple=True)
@click.option("-id", "--in_dataset", type=str, multiple=True)
@click.option("--min-raw", type=float, multiple=True, default=[0,])
@click.option("--max-raw", type=float, multiple=True, default=[255,])
@click.option(
    "-mc",
    "--mask-container",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)
@click.option(
    "-md",
    "--mask-dataset",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)
@click.option(
    "--instance",
    type=bool,
    default=False,
)
def start_worker(
    setup_dir,
    checkpoint,
    input_tensor,
    output_tensor,
    num_outputs,
    channels,
    voxel_size,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    min_raw,
    max_raw,
    mask_container,
    mask_dataset,
    instance,
):
    with open(os.path.join(setup_dir, "config.json"), "r") as f:
        net_config = json.load(f)
        output_tensorname = []
        input_tensorname = []
        for ot in output_tensor:
            output_tensorname.append(net_config[ot])
        for it in input_tensor:
            input_tensorname.append(net_config[it])

    # voxels
    # input_shape = Coordinate(net_config['input_shape'])
    # output_shape = Coordinate(net_config['output_shape'])
    shifts = []
    scales = []
    for minr, maxr in zip(min_raw, max_raw):
        shifts.append(minr)
        scales.append(maxr - minr)
    assert len(channels) == len(output_tensor)
    parsed_channels = []
    for channel_split in channels:
        parsed_channels.append(
            list(channel.split(":") for channel in channel_split.split(","))
        )

    client = daisy.Client()
    session = load_eval_model(setup_dir, checkpoint)
    input_dataset = []
    for in_container_iter, in_dataset_iter in zip(in_container, in_dataset):
        input_dataset.append(
            open_ds(os.path.join(in_container_iter, in_dataset_iter), voxel_size=voxel_size)
        )
    mask_datasets = [
        open_ds(os.path.join(mc, md)) for mc, md in zip(mask_container, mask_dataset)
    ]

    # voxel_size = raw_dataset.voxel_size
    output_voxel_size = daisy.Coordinate(voxel_size)
    num_channels = num_outputs
    if not instance:
        out_datasets = []
        for channel in parsed_channels:
            out_datasets.append([])
            for _, ch in channel:
                out_datasets[-1].append(
                    open_ds(
                        os.path.join(out_container, f"{out_dataset}/{ch}"),
                        mode="r+",
                    )
                )
    else:
        raise NotImplementedError()
        # assert len(parsed_channels) == 1
        # indexes, channel = parsed_channels[0]
        # out_datasets = [
        #     open_ds(os.path.join(out_container, f"{out_dataset}/{channel}__{i}"), mode="r+") for i in range(0, num_channels, 3)
        # ]

    while True:
        with client.acquire_block() as block:
            if block is None:
                break
            if len(mask_datasets) > 0:
                mask_data = any(
                    [
                        np.any(
                            mask_dataset.to_ndarray(
                                roi=block.read_roi.snap_to_grid(
                                    mask_dataset.voxel_size
                                ),
                                fill_value=0,
                            )
                        )
                        for mask_dataset in mask_datasets
                    ]
                )
            else:
                mask_data = 1
            if not np.any(mask_data):
                # avoid predicting if mask is empty
                continue
            inputs = []
            for in_dataset_iter, shift, scale in zip(input_dataset, shifts, scales):
                inputs.append(
                    (2.0
                    * (
                        in_dataset_iter.to_ndarray(
                            roi=block.read_roi, fill_value=shift + scale
                        ).astype(np.float32)
                        - shift
                    )
                    / scale) - 1.0
                )
            # raw_input = np.expand_dims(raw_input, (0, 1))
            print(block)
            write_roi = block.write_roi.intersect(out_datasets[0][0].roi)

            if out_datasets[0][0].to_ndarray(write_roi).any():
                # block has already been processed
                continue
            output_data = session.run(
                {ot: ot for ot in output_tensorname},
                feed_dict={k:v for k, v in zip(input_tensorname, inputs)},
            )
            print(f"{output_tensorname=}, {parsed_channels=}, {out_datasets=}")
            for ot, channel_split, od in zip(
                output_tensorname, parsed_channels, out_datasets
            ):
                predictions = Array(
                    output_data[ot],
                    block.write_roi.offset,
                    output_voxel_size,
                    axis_names=["c^", "z", "y", "x"],
                )

                write_data = predictions.to_ndarray(write_roi).clip(0, 1)
                # if not instance:
                write_data = (write_data) * 255.0  # / 2.0

                for (i, _), od_iter in zip(channel_split, od):
                    indexes = []
                    if "-" in i:
                        j, k = i.split("-")
                        indexes = list(range(int(j), int(k)))
                    else:
                        indexes = [int(i)]
                    print(f"{indexes=}")
                    print(f"{od_iter=}")
                    if len(indexes) > 1:
                        # print(f"{out_dataset[write_roi]}")
                        od_iter[write_roi] = np.round(
                            np.stack([write_data[j] for j in indexes], axis=0)
                        ).astype(np.uint8)
                    else:
                        od_iter[write_roi] = write_data[indexes[0]].astype(np.uint8)
                    # else:
                #     for i, out_dataset in zip(range(0, num_channels, 3), out_datasets):
                #         out_dataset[write_roi] = write_data[i : i + 3].astype(np.float32)

            block.status = daisy.BlockStatus.SUCCESS


if __name__ == "__main__":
    cli()
    # load_eval_model("/nrs/saalfeld/heinrichl/fly_organelles/lsd/networks/fib25/mtlsd/", "train_net_checkpoint_400000")
    # input_tensor = "raw"
    # output_tensor="embedding"
