import tensorflow_hub as hub
import tensorflow as tf


def import_module(path='/ghome/fengrl/compare_gan/bigbigan'):
    m = hub.Module(path)
    return m


def make_prune(signature='gen'):
    m = import_module()
    def model_fn():
        z = tf.placeholder(dtype=tf.float32)
        x = m(z, signature='generate', as_dict=True)['default']
        hub.add_signature(signature, z, x)
    tags_and_args = [({'train'}, {}), (set(), {})]
    drop_collections = [v for v in tf.global_variables() if 'Generator' not in v.name]
    p = hub.create_module_spec(model_fn, tags_and_args, drop_collections)
    return p