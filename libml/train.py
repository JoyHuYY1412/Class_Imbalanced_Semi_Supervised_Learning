# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training loop, checkpoint saving and loading, evaluation code."""
import functools
import json
import os.path
import shutil

import numpy as np
import tensorflow as tf
from absl import flags
from tqdm import trange, tqdm

from libml import data, utils
from libml.utils import EasyDict

FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './experiments',
                    'Folder where to save training data.')
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_integer('train_kimg', 1 << 14, 'Training duration in kibi-samples.')
flags.DEFINE_integer('report_kimg', 64, 'Report summary period in kibi-samples.')
flags.DEFINE_integer('save_kimg', 64, 'Save checkpoint period in kibi-samples.')
flags.DEFINE_integer('keep_ckpt', 50, 'Number of checkpoints to keep.')
flags.DEFINE_string('eval_ckpt', '', 'Checkpoint to evaluate. If provided, do not do training, just do eval.')
flags.DEFINE_string('rerun', '', 'A string to identify a run if running multiple ones with same parameters.')


class Model:
    def __init__(self, train_dir: str, dataset: data.DataSets, **kwargs):
        self.train_dir = os.path.join(train_dir, FLAGS.rerun, self.experiment_name(**kwargs))
        self.params = EasyDict(kwargs)
        self.dataset = dataset
        self.session = None
        self.tmp_p = np.ones(dataset.nclass)/dataset.nclass
        # self.p_cls_ema = tf.Variable(tf.ones(dataset.nclass)/dataset.nclass)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        # self.p_cls_ema = np.ones(dataset.nclass)/dataset.nclass
        self.tmp = EasyDict(print_queue=[], cache=EasyDict())
        self.step = tf.train.get_or_create_global_step()
        self.ops = self.model(**kwargs)
        self.ops.update_step = tf.assign_add(self.step, FLAGS.batch)
        # self.ops.p_cls_ema = tf.assign_add(self.p_cls_ema, self.p_cls_ema)

        self.add_summaries(**kwargs)

        print(' Config '.center(80, '-'))
        print('train_dir', self.train_dir)
        print('%-32s %s' % ('Model', self.__class__.__name__))
        print('%-32s %s' % ('Dataset', dataset.name))
        for k, v in sorted(kwargs.items()):
            print('%-32s %s' % (k, v))
        print(' Model '.center(80, '-'))
        to_print = [tuple(['%s' % x for x in (v.name, np.prod(v.shape), v.shape)]) for v in utils.model_vars(None)]
        to_print.append(('Total', str(sum(int(x[1]) for x in to_print)), ''))
        sizes = [max([len(x[i]) for x in to_print]) for i in range(3)]
        fmt = '%%-%ds  %%%ds  %%%ds' % tuple(sizes)
        for x in to_print[:-1]:
            print(fmt % x)
        print()
        print(fmt % to_print[-1])
        print('-' * 80)
        self._create_initial_files()

    @property
    def arg_dir(self):
        return os.path.join(self.train_dir, 'args')

    @property
    def checkpoint_dir(self):
        return os.path.join(self.train_dir, 'tf')

    def train_print(self, text):
        self.tmp.print_queue.append(text)

    def _create_initial_files(self):
        for dir in (self.checkpoint_dir, self.arg_dir):
            tf.gfile.MakeDirs(dir)
        self.save_args()

    def _reset_files(self):
        shutil.rmtree(self.train_dir)
        self._create_initial_files()

    def save_args(self, **extra_params):
        with tf.gfile.Open(os.path.join(self.arg_dir, 'args.json'), 'w') as f:
            json.dump({**self.params, **extra_params}, f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, train_dir):
        with tf.gfile.Open(os.path.join(train_dir, 'args/args.json'), 'r') as f:
            params = json.load(f)
        instance = cls(train_dir=train_dir, **params)
        instance.train_dir = train_dir
        return instance

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items())]
        return '_'.join([self.__class__.__name__] + args)

    def eval_mode(self, ckpt=None):
        self.session = tf.Session(config=utils.get_config())
        saver = tf.train.Saver()
        if ckpt is None:
            ckpt = utils.find_latest_checkpoint(self.checkpoint_dir)
        else:
            ckpt = os.path.abspath(ckpt)
        saver.restore(self.session, ckpt)
        self.tmp.step = self.session.run(self.step)
        print('Eval model %s at global_step %d' % (self.__class__.__name__, self.tmp.step))
        return self

    def model(self, **kwargs):
        raise NotImplementedError()

    def add_summaries(self, **kwargs):
        raise NotImplementedError()


class ClassifySemi(Model):
    """Semi-supervised classification."""

    def __init__(self, train_dir: str, dataset: data.DataSets, nclass: int, **kwargs):
        self.nclass = nclass

        Model.__init__(self, train_dir, dataset, nclass=nclass, **kwargs)

    def train_step(self, train_session, gen_labeled, gen_unlabeled):
        x, y = gen_labeled(), gen_unlabeled()
        # self.tmp.step = train_session.run([self.ops.train_op, self.ops.update_step],
        #                                   feed_dict={self.ops.y: y['image'],
        #                                              self.ops.xt: x['image'],
        #                                              self.ops.label: x['label']})[1]

        v = train_session.run([self.ops.train_op, self.ops.update_step, self.ops.update_p],
                              feed_dict={self.ops.y: y['image'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label'],
                                         self.ops.p_cls_ema: self.tmp_p})
        self.tmp_p=v[-1]
        # self.p_cls_ema = tf.convert_to_tensor(v[-1])
        self.tmp.step = v[-2]

    def gen_labeled_fn(self, data_iterator):
        return self.dataset.train_labeled.numpy_augment(lambda: self.session.run(data_iterator))

    def gen_unlabeled_fn(self, data_iterator):
        return self.dataset.train_unlabeled.numpy_augment(lambda: self.session.run(data_iterator))

    def train(self, train_nimg, report_nimg):
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt, pad_step_number=10))

        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            gen_labeled = self.gen_labeled_fn(train_labeled)
            gen_unlabeled = self.gen_unlabeled_fn(train_unlabeled)
            self.tmp.step = self.session.run(self.step)
            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                for _ in loop:
                    self.train_step(train_session, gen_labeled, gen_unlabeled)
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))

    def eval_checkpoint(self, ckpt=None):
        self.eval_mode(ckpt)
        self.cache_eval()
        raw = self.eval_stats(classify_op=self.ops.classify_raw)
        ema = self.eval_stats(classify_op=self.ops.classify_op)
        print('%16s %8s %8s %8s' % ('', 'labeled', 'valid', 'test'))
        print('%16s %8s %8s %8s' % (('raw',) + tuple('%.2f' % x for x in raw)))
        print('%16s %8s %8s %8s' % (('ema',) + tuple('%.2f' % x for x in ema)))

    def cache_eval(self):
        """Cache datasets for computing eval stats."""

        def collect_samples(dataset, name):
            """Return numpy arrays of all the samples from a dataset."""
            pbar = tqdm(desc='Caching %s examples' % name)
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            images, labels = [], []
            while 1:
                try:
                    v = self.session.run(it)
                except tf.errors.OutOfRangeError:
                    break
                images.append(v['image'])
                labels.append(v['label'])
                pbar.update()

            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            pbar.close()
            return images, labels

        if 'test' not in self.tmp.cache:
            self.tmp.cache.test = collect_samples(self.dataset.test.parse(), name='test')
            self.tmp.cache.valid = collect_samples(self.dataset.valid.parse(), name='valid')
            self.tmp.cache.train_labeled = collect_samples(self.dataset.train_labeled.take(10000).parse(),
                                                           name='train_labeled')
            self.tmp.cache.train_unlabeled = collect_samples(self.dataset.train_unlabeled.parse(),
                                                           name='train_unlabeled')
    def eval_stats(self, batch=None, feed_extra=None, classify_op=None, verbose=True):
        """Evaluate model on train, valid and test."""
        batch = batch or FLAGS.batch
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        accuracies = []
        for subset in ('train_labeled', 'valid', 'test'):
            images, labels = self.tmp.cache[subset]

            if subset == 'test':
                num_class = self.dataset.nclass
                classwise_num = np.zeros(num_class)
                classwise_correct = np.zeros(num_class)
                for lab_i in labels:
                    classwise_num[lab_i] = classwise_num[lab_i] + 1
                assert np.sum(classwise_num) == labels.shape[0]

            predicted = []

            for x in range(0, images.shape[0], batch):
                p = self.session.run(
                    classify_op,
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })
                predicted.append(p)
            predicted = np.concatenate(predicted, axis=0)
            predicted_label = predicted.argmax(1)
            predicted_confidence = predicted.max(1)

            if subset == 'valid':
                vaild_info = dict()
                vaild_info['labels'] = labels
                vaild_info['condience'] = predicted_confidence
                vaild_info['predict'] = predicted_label
                import json
                with open("/gruntdata2/xinting/project/tf_fixmatch/experiments/fixmatch/cifar10_LT_50.d.d.d.1@50-50000/CTAugment_depth2_th0.80_decay0.990/FixMatch_archresnet_batch64_confidence0.95_db0_devicenum1_filters32_lr0.03_nclass10_repeat4_scales3_train_kimg8192_upper10.0_uratio7_wd0.0005_weight_l_ce0_weight_ulb0_wu1.0/prediction.json","w") as f:
                    json.dump(vaild_info, f)
                import pdb; pdb.set_trace() 

            del predicted
            
            if subset == 'test':
                for image_i in range(images.shape[0]):
                    if predicted_label[image_i] == labels[image_i]:
                        classwise_correct[predicted_label[image_i]] += 1

            accuracies.append((predicted_label == labels).mean() * 100)

            if subset == 'test':
                # claculate geometric mean
                classwise_acc = (classwise_correct / classwise_num)
                GM = 1
                for i in range(num_class):
                    if classwise_acc[i] == 0:
                        # To prevent the N/A values, we set the minimum value as 0.001
                        GM *= (1/(100 * num_class)) ** (1/num_class)
                    else:
                        GM *= (classwise_acc[i]) ** (1/num_class)
                accuracies.append(GM * 100)
                # accuracy per class
                accuracies.extend(classwise_acc * 100)

        if verbose:
            self.train_print('kimg %-5d  accuracy train/valid/test/GM  %.2f  %.2f  %.2f %.2f' %
                             tuple([self.tmp.step >> 10] + accuracies[:4]))
        return np.array(accuracies, 'f')

    def add_summaries(self, feed_extra=None, **kwargs):
        del kwargs

        def gen_stats(classify_op=None, verbose=True):
            return self.eval_stats(feed_extra=feed_extra, classify_op=classify_op, verbose=verbose)

        accuracies = tf.py_func(functools.partial(gen_stats), [], tf.float32)
        tf.summary.scalar('accuracy/train_labeled', accuracies[0])
        tf.summary.scalar('accuracy/valid', accuracies[1])
        tf.summary.scalar('accuracy', accuracies[2])
        tf.summary.scalar('accuracy/GM', accuracies[3])

        for cls_i in range(self.dataset.nclass):
            tf.summary.scalar('accuracy_cls/cls'+str(cls_i), accuracies[4+cls_i])

        if 'classify_raw' in self.ops:
            accuracies = tf.py_func(functools.partial(gen_stats,
                                                      classify_op=self.ops.classify_raw,
                                                      verbose=False), [], tf.float32)
            tf.summary.scalar('accuracy/raw/train_labeled', accuracies[0])
            tf.summary.scalar('accuracy/raw/valid', accuracies[1])
            tf.summary.scalar('accuracy/raw', accuracies[2])
