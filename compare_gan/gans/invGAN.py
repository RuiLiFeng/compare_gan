from compare_gan.gans import modular_gan
from compare_gan.architectures import invert
from functools import partial
import tensorflow as tf


@gin.configurable(blacklist=["dataset", "parameters", "model_dir"])
class invGAN(modular_gan.ModularGAN):
    def __init__(self,
                 dataset,
                 parameters,
                 model_dir,
                 deprecated_split_disc_calls=False,
                 experimental_joint_gen_for_disc=False,
                 experimental_force_graph_unroll=False,
                 g_use_ema=False,
                 ema_decay=0.9999,
                 ema_start_step=40000,
                 g_optimizer_fn=tf.train.AdamOptimizer,
                 d_optimizer_fn=None,
                 g_lr=0.0002,
                 d_lr=None,
                 conditional=False,
                 fit_label_distribution=False):
        super(invGAN, self).__init__(dataset, parameters, model_dir, deprecated_split_disc_calls,
                                     experimental_joint_gen_for_disc, experimental_force_graph_unroll,
                                     g_use_ema, ema_decay, ema_start_step, g_optimizer_fn,
                                     d_optimizer_fn, g_lr, d_lr, conditional, fit_label_distribution)

    @property
    def generator(self):
        if self._generator is None:
            self._generator = invert.Generator(image_shape=self._dataset.image_shape)
        return self._generator
