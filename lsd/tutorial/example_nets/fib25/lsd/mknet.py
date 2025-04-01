import json
import os

import click
import tensorflow.compat.v1 as tf

from lsd.networks import conv_pass, unet

tf.disable_eager_execution()
os.environ["TF_USE_LEGACY_KERAS"] = "1"


def create_network(input_shape, name):

    tf.reset_default_graph()

    with tf.variable_scope("setup03"):
        raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(raw, (1, 1) + input_shape)

        model, _, _ = unet(raw_batched, 12, 6, [[2, 2, 2], [2, 2, 2], [3, 3, 3]])

        embedding_batched, _ = conv_pass(
            model,
            kernel_sizes=[1],
            num_fmaps=10,
            activation="sigmoid",
            name="embedding",
        )

        output_shape_batched = embedding_batched.get_shape().as_list()
        output_shape = output_shape_batched[1:]  # strip the batch dimension

        embedding = tf.reshape(embedding_batched, output_shape)

        gt_embedding = tf.placeholder(tf.float32, shape=output_shape)
        loss_weights_embedding = tf.placeholder(tf.float32, shape=output_shape)

        loss = tf.losses.mean_squared_error(
            gt_embedding, embedding, loss_weights_embedding
        )

        summary = tf.summary.scalar("lsd_eucl_loss", loss)

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
        )
        optimizer = opt.minimize(loss)

        output_shape = output_shape[1:]

        print("input shape : %s" % (input_shape,))
        print("output shape: %s" % (output_shape,))

        tf.train.export_meta_graph(filename=name + ".meta")

        config = {
            "raw": raw.name,
            "embedding": embedding.name,
            "gt_embedding": gt_embedding.name,
            "loss_weights_embedding": loss_weights_embedding.name,
            "loss": loss.name,
            "optimizer": optimizer.name,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "summary": summary.name,
        }

        config["outputs"] = {"lsds": {"out_dims": 10, "out_dtype": "uint8"}}

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
    "--steps",
    type=int,
    default=20,
    help="number of steps by which to increase shape in xy dimension (step size: 27)",
)
def cli(output_config, steps):
    size = steps * 2*3*3 + 124
    create_network((size, size, size), output_config)


if __name__ == "__main__":
    cli()
