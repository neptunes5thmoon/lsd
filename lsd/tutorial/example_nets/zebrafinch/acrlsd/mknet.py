import json
import os

import click
import tensorflow.compat.v1 as tf

from lsd.networks import conv_pass, crop_zyx, unet

tf.disable_eager_execution()
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# def create_auto(input_shape, output_shape, name):
#     tf.reset_default_graph()

#     with tf.variable_scope("setup03"):
#         raw = tf.placeholder(tf.float32, shape=input_shape)
#         raw_batched = tf.reshape(raw, (1, 1) + input_shape)

#         unet, _, _ = mala.networks.unet(
#             raw_batched, 12, 5, [[1, 3, 3], [1, 3, 3], [3, 3, 3]]
#         )

#         embedding_batched, _ = mala.networks.conv_pass(
#             unet, kernel_sizes=[1], num_fmaps=10, activation="sigmoid", name="embedding"
#         )

#         embedding_batched = crop_zyx(embedding_batched, (1, 10) + output_shape)
#         embedding = tf.reshape(embedding_batched, (10,) + output_shape)

#         print("input shape : %s" % (input_shape,))
#         print("output shape: %s" % (output_shape,))

#         tf.train.export_meta_graph(filename=name + ".meta")

#         config = {
#             "raw": raw.name,
#             "embedding": embedding.name,
#             "input_shape": input_shape,
#             "output_shape": output_shape,
#         }
#         with open(name + ".json", "w") as f:
#             json.dump(config, f)


def create_affs(input_shape, intermediate_shape, expected_output_shape, name):
    tf.reset_default_graph()

    with tf.variable_scope("setup05"):
        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)
        raw_in = tf.reshape(raw_batched, input_shape)
        raw_batched = crop_zyx(raw_batched, (1, 1) + intermediate_shape)
        raw_cropped = tf.reshape(raw_batched, intermediate_shape)

        pretrained_lsd = tf.placeholder(tf.float32, shape=(10,) + intermediate_shape)
        pretrained_lsd_batched = tf.reshape(
            pretrained_lsd, (1, 10) + intermediate_shape
        )

        concat_input = tf.concat([raw_batched, pretrained_lsd_batched], axis=1)

        model, _, _ = unet(concat_input, 12, 5, [[1, 3, 3], [1, 3, 3], [3, 3, 3]])

        affs_batched, _ = conv_pass(
            model, kernel_sizes=[1], num_fmaps=3, activation="sigmoid", name="affs"
        )
        affs = tf.squeeze(affs_batched, axis=0)

        output_shape = tuple(affs.get_shape().as_list()[1:])
        assert expected_output_shape == output_shape, "%s !=%s" % (
            expected_output_shape,
            output_shape,
        )

        gt_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)
        loss_weights_affs = tf.placeholder(tf.float32, shape=(3,) + output_shape)

        loss = tf.losses.mean_squared_error(gt_affs, affs, loss_weights_affs)

        summary = tf.summary.scalar("setup05_eucl_loss", loss)

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
        )
        optimizer = opt.minimize(loss)

        print("input shape : %s" % (intermediate_shape,))
        print("output shape: %s" % (output_shape,))

        tf.train.export_meta_graph(filename=name + ".meta")

        config = {
            "raw": raw.name,
            "raw_in": raw_in.name,
            "pretrained_lsd": pretrained_lsd.name,
            "affs": affs.name,
            "gt_affs": gt_affs.name,
            "loss_weights_affs": loss_weights_affs.name,
            "loss": loss.name,
            "optimizer": optimizer.name,
            "input_shape": intermediate_shape,
            "output_shape": output_shape,
            "summary": summary.name,
        }
        with open(name + ".json", "w") as f:
            json.dump(config, f)


def create_config(input_shape, output_shape, name):
    config = {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "lsds_setup": "lsd",
        "lsds_iteration": 400000,
    }

    config["outputs"] = {"affs": {"out_dims": 3, "out_dtype": "uint8"}}

    with open(name + ".json", "w") as f:
        json.dump(config, f)


@click.command()
@click.option(
    "--output-config",
    type=click.Path(),
    required=True,
    help="Directory to save the train_net and config files. Will save files under that name with .json and .meta extensions.",
)
@click.option(
    "--z",
    type=int,
    default=19,
    help="number of steps by which to increase shape in z dimension (step size: 3)",
)
@click.option(
    "--xy",
    type=int,
    default=10,
    help="number of steps by which to increase shape in xy dimension (step size: 27)",
)
def cli(output_config, z, xy):
    input_shape = (96 + z, 484 + xy, 484+xy)
    output_shape = (60 + z, 272 + xy, 484 + xy)
    create_affs(input_shape, input_shape, output_shape, output_config)


if __name__ == "__main__":
    z = 0
    xy = 0

    # train_input_shape = (120, 484, 484)
    # train_intermediate_shape = (84, 268, 268)
    # train_output_shape = (48, 56, 56)

    # create_auto(train_input_shape, train_intermediate_shape, "train_auto_net")
    # create_affs(
    # train_input_shape, train_intermediate_shape, train_output_shape, "train_net"
    # )

    test_input_shape = (96 + z, 484 + xy, 484 + xy)
    test_output_shape = (60 + z, 272 + xy, 272 + xy)

    create_affs(
        test_input_shape,
        test_input_shape,
        test_output_shape,
        "/nrs/saalfeld/heinrichl/fly_organelles/lsd/networks/zebrafinch/acrlsd/config",
    )

    # create_config(test_input_shape, test_output_shape, "config")
