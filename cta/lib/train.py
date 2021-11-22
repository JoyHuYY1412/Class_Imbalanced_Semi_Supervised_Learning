
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

import numpy as np
from absl import flags

from fully_supervised.lib.train import ClassifyFullySupervised
from libml import data
from libml.augment import AugmentPoolCTA
from libml.ctaugment import CTAugment
from libml.train import ClassifySemi
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer('adepth', 2, 'Augmentation depth.')
flags.DEFINE_float('adecay', 0.99, 'Augmentation decay.')
flags.DEFINE_float('ath', 0.80, 'Augmentation threshold.')


class CTAClassifySemi(ClassifySemi):
    """Semi-supervised classification."""
    AUGMENTER_CLASS = CTAugment
    AUGMENT_POOL_CLASS = AugmentPoolCTA

    @classmethod
    def cta_name(cls):
        return '%s_depth%d_th%.2f_decay%.3f' % (cls.AUGMENTER_CLASS.__name__,
                                                FLAGS.adepth, FLAGS.ath, FLAGS.adecay)

    def __init__(self, train_dir: str, dataset: data.DataSets, nclass: int, **kwargs):
        ClassifySemi.__init__(self, train_dir, dataset, nclass, **kwargs)
        self.augmenter = self.AUGMENTER_CLASS(FLAGS.adepth, FLAGS.ath, FLAGS.adecay)

    def gen_labeled_fn(self, data_iterator):
        def wrap():
            batch = self.session.run(data_iterator)
            batch['cta'] = self.augmenter
            batch['probe'] = True
            return batch

        return self.AUGMENT_POOL_CLASS(wrap)

    def gen_unlabeled_fn(self, data_iterator):
        def wrap():
            batch = self.session.run(data_iterator)
            batch['cta'] = self.augmenter
            batch['probe'] = False
            return batch

        return self.AUGMENT_POOL_CLASS(wrap)

    def train_step(self, train_session, gen_labeled, gen_unlabeled):
        x, y = gen_labeled(), gen_unlabeled()

        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.update_step, self.ops.update_p],
                              feed_dict={self.ops.y: y['image'],
                                         self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label'],
                                         self.ops.p_cls_ema: self.tmp_p})
        self.tmp_p=v[-1]
        # self.p_cls_ema = tf.convert_to_tensor(v[-1])
        self.tmp.step = v[-2]
        lx = v[0]
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)

    def eval_stats(self, batch=None, feed_extra=None, classify_op=None, verbose=True):
        """Evaluate model on train, valid and test."""
        '''
        batch = batch or FLAGS.batch
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        accuracies = []
        for subset in ('train_labeled', 'valid', 'test'):
            images, labels = self.tmp.cache[subset]
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
            accuracies.append((predicted.argmax(1) == labels).mean() * 100)
        if verbose:
            self.train_print('kimg %-5d  accuracy train/valid/test  %.2f  %.2f  %.2f' %
                             tuple([self.tmp.step >> 10] + accuracies))
        self.train_print(self.augmenter.stats())
        return np.array(accuracies, 'f')
        '''
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
            
            # if subset == 'valid':
            #     # for boxplot only
            #     predicted_confidence = predicted.max(1)
            #     a = np.array(range(len(predicted)))
            #     b = labels
            #     true_confidence = predicted[a, b]
            #     vaild_info = dict()
            #     vaild_info['labels'] = labels.tolist()
            #     vaild_info['confidence'] = predicted_confidence.tolist()
            #     vaild_info['true_confidence'] = true_confidence.tolist()
            #     vaild_info['predict'] = predicted_label.tolist()
            #     import pickle
            #     a_file = open("/gruntdata2/xinting/project/tf_fixmatch/experiments/fixmatch/cifar10_LT_50.d.d.d.1@50-50000/CTAugment_depth2_th0.80_decay0.990/data_0_1.pkl", "wb")
            #     pickle.dump(vaild_info, a_file)
            #     a_file.close()
            #     import pdb; pdb.set_trace() 

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
        # self.train_print(self.augmenter.stats())
        
        return np.array(accuracies, 'f')


class CTAClassifyFullySupervised(ClassifyFullySupervised, CTAClassifySemi):
    """Fully-supervised classification."""

    def train_step(self, train_session, gen_labeled):
        x = gen_labeled()
        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.update_step],
                              feed_dict={self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label']})
        self.tmp.step = v[-1]
        lx = v[0]
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)
