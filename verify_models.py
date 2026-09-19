"""Static model-graph verification for the three NTU scripts.

Builds every stream and every fusion model *without* touching the dataset and
without training, then prints the parameter counts and output shapes. Use it to
confirm that the refactor did not change the architecture, and that the C3D
branch now accepts the 3-channel SSIs produced by `MetricLayer_ForC3D`.

    python verify_models.py

Requires only Keras + a backend; no .mat / .h5 data and no GPU.
Writes nothing.
"""

import imp
import sys
import traceback

import keras
from keras import regularizers
from keras.layers import Input, Lambda
from keras.models import Model


def _load_by_path(script_name):
    """Import one of the hyphenated training scripts as a module.

    The filenames contain '-', so they cannot be imported normally. Each
    script guards its body with `if __name__ == '__main__'`, so loading it
    here defines the helpers without running any training.
    """
    mod_name = script_name.replace('-', '_').replace('.py', '')
    return imp.load_source(mod_name, script_name)


def _describe(model, label):
    total = model.count_params()
    trainable = sum(int(w.shape.num_elements() or 0)
                    for w in model.trainable_weights)
    print('  %-28s params=%-12d trainable=%-12d out=%s'
          % (label, total, trainable, model.output_shape))
    return total


def check_late():
    print('\n=== ntu-latefusion-spp-metric.py ===')
    m = _load_by_path('ntu-latefusion-spp-metric.py')
    reg = regularizers.l2(m.LAMBDA1)
    stride = [1, 1, 1]
    specs = [
        ('1', 7 * 2, [4], [7], 16),
        ('2', 12 * 2, [4], [12], 32),
        ('3', 25 * 2, [4], [20], 64),
    ]
    total = 0
    for scale, jn, spp, spp_a, filters in specs:
        layers_num = [jn, filters // 4, filters // 2, filters, 100, 100,
                      sum(x ** 2 for x in spp) * filters]
        model = m.get_model(layers_num, jn, m.MAX_LEN, stride, m.CLASS_NUM,
                            reg, spp, spp_a, scale)
        total += _describe(model, 'stream scale=%s' % scale)
        # the metric layer must be present and single-channel
        ml = model.get_layer('metric_layer-' + scale)
        assert ml.output_shape[-1] == 1, ml.output_shape
    print('  total across streams: %d' % total)


def check_early():
    print('\n=== ntu-earlyfusion-spp-metric.py ===')
    m = _load_by_path('ntu-earlyfusion-spp-metric.py')
    reg = regularizers.l2(m.LAMBDA1)
    stride = [1, 1, 1]
    specs = [
        ('1', 7 * 2, [4], [7], 16),
        ('2', 12 * 2, [4], [12], 32),
        ('3', 25 * 2, [4], [20], 64),
    ]
    for scale, jn, spp, spp_a, filters in specs:
        layers_num = [jn, filters // 4, filters // 2, filters, 100,
                      sum(x ** 2 for x in spp) * filters]
        model = m.get_model(layers_num, jn, m.MAX_LEN, stride, m.CLASS_NUM,
                            reg, spp, spp_a, scale)
        _describe(model, 'stream scale=%s' % scale)
        ml = model.get_layer('metric_layer-' + scale)
        assert ml.output_shape[-1] == 1, ml.output_shape


def check_c3d():
    """The important one: does the C3D backbone accept the SSI now?"""
    print('\n=== ntu-earlyfusion-spp-metric-c3d.py ===')
    m = _load_by_path('ntu-earlyfusion-spp-metric-c3d.py')
    reg = regularizers.l2(m.LAMBDA1)

    for scale, jn in (('1', 7 * 2), ('2', 12 * 2), ('3', 25 * 2)):
        main_input = Input(shape=(m.MAX_LEN, jn, 3), dtype='float32')
        r1 = Lambda(m.repeat_x_onejoint,
                    m.repeat_x_onejoint_output_shape)(main_input)
        r2 = Lambda(m.repeat_x_groupjoint,
                    m.repeat_x_groupjoint_output_shape)(main_input)
        ssm = keras.layers.subtract([r1, r2])
        ssm = m.MetricLayer_ForC3D(jn, kernel_regularizer=reg,
                                   name='metric_layer-' + scale)(ssm)
        ssm = Lambda(m.flattenSSM3, m.flattenSSM3_output_shape)(ssm)
        probe = Model(inputs=[main_input], outputs=[ssm])
        ch = probe.output_shape[-1]
        status = 'OK' if ch == 3 else 'WRONG'
        print('  scale=%s SSI out=%s channels=%d  %s'
              % (scale, probe.output_shape, ch, status))
        assert ch == 3, 'C3D branch needs 3 channels, got %d' % ch

    print('  -> SSI is 3-channel; the Sports-1M conv1 weights will load')
    print('     without a shape mismatch.')
    print('  NOTE: this is the change that can move MSNN_early-C3D numbers.')


def main():
    failures = []
    for fn in (check_late, check_early, check_c3d):
        try:
            fn()
        except Exception:
            failures.append(fn.__name__)
            traceback.print_exc()
    print('\n' + '=' * 60)
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        return 1
    print('All model graphs build. Compare the parameter counts above against')
    print('TABLE II of the paper, or against a pre-refactor checkout.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
