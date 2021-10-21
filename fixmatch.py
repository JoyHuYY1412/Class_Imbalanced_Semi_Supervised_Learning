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
import functools
import os

import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from tqdm import trange

from cta.cta_remixmatch import CTAReMixMatch
from libml import data, utils, augment, ctaugment
FLAGS = flags.FLAGS


class AugmentPoolCTACutOut(augment.AugmentPoolCTA):
    @staticmethod
    def numpy_apply_policies(arglist):
        x, cta, probe = arglist
        if x.ndim == 3:
            assert probe
            policy = cta.policy(probe=True)
            return dict(policy=policy,
                        probe=ctaugment.apply(x, policy),
                        image=x)
        assert not probe
        cutout_policy = lambda: cta.policy(probe=False) + [ctaugment.OP('cutout', (1,))]
        return dict(image=np.stack([x[0]] + [ctaugment.apply(y, cutout_policy()) for y in x[1:]]).astype('f'))


class FixMatch(CTAReMixMatch):
    AUGMENT_POOL_CLASS = AugmentPoolCTACutOut

    def train(self, train_nimg, report_nimg):

        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(batch * self.params['uratio']).prefetch(16)
        train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))

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

    def model(self, batch, lr, wd, wu, confidence, uratio, ema=0.999, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # Training labeled
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')  # Eval images
        y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')  # Training unlabeled (weak, strong)
        l_in = tf.placeholder(tf.int32, [batch], 'labels')  # Labels
        p_in = tf.placeholder(tf.float32, self.dataset.nclass, 'p_cls_ema')  # Labels

        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        # Compute logits for xt_in and y_in
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        x = utils.interleave(tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0), 2 * uratio + 1)
        logits = utils.para_cat(lambda x: classifier(x, training=True), x)
        logits = utils.de_interleave(logits, 2 * uratio+1)

        # # compute logits for x
        # if FLAGS.db > 0:
        #     logits_aug_strong = utils.para_cat(lambda x: classifier(x, training=True), x_in)

        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        logits_x = logits[:batch]
        logits_weak, logits_strong = tf.split(logits[batch:], 2)
        del logits, skip_ops

        # Labeled cross-entropy
        if (FLAGS.weight_l_ce == 0) or (FLAGS.weight_l_ce == 6) or (FLAGS.weight_l_ce == 7):
            if FLAGS.weight_l_ce == 0:
                loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l_in, logits=logits_x)
                loss_xe = tf.reduce_mean(loss_xe)
                # update_p = tf.add(1.00*p_in, 0.0)
                p_each_cls = get_p_cls(output_lb=logits_x, targets=l_in, output_ulb=logits_weak, weight_l_ce=FLAGS.weight_l_ce)
                update_p = tf.add(0.99*p_in, 0.01*p_each_cls)
            elif FLAGS.weight_l_ce == 6:
                # re-weighting
                loss_xe = reweight_ce_loss(output_lb=logits_x, targets=l_in)
                loss_xe = tf.reduce_mean(loss_xe)
                update_p = tf.add(1.00*p_in, 0.0)  
            elif FLAGS.weight_l_ce == 7:
                # re-weighting
                loss_xe = la_ce_loss(output_lb=logits_x, targets=l_in)
                loss_xe = tf.reduce_mean(loss_xe)
                update_p = tf.add(1.00*p_in, 0.0)  
        else:
            if FLAGS.db == 2:
                loss_xe, loss_xe_aug, p_each_cls = weighted_ce_loss_aug(output_lb=logits_x, targets=l_in, output_ulb=logits_weak, weight_l_ce=FLAGS.weight_l_ce, p_cls_ema=p_in, 
                step=self.step, upper=FLAGS.upper)
            else:
                loss_xe, p_each_cls = weighted_ce_loss(output_lb=logits_x, targets=l_in, output_ulb=logits_weak, weight_l_ce=FLAGS.weight_l_ce, p_cls_ema=p_in, 
                step=self.step, upper=FLAGS.upper)
            # print(self.tmp.step)
            update_p = tf.add(0.90*p_in, 0.1*p_each_cls)

            loss_xe = tf.reduce_mean(loss_xe)
            if FLAGS.db == 2:
                pseudo_labels_aug = tf.stop_gradient(tf.nn.softmax(logits_x))
                thres = confidence * (update_p/tf.reduce_max(update_p))**0.5
                max_idx = tf.argmax(pseudo_labels_aug, axis=1)
                max_probs = tf.reduce_max(pseudo_labels_aug, axis=1)
                pseudo_mask = tf.zeros_like(max_probs)
                for cls_i in range(self.dataset.nclass):
                    mask_cls_i = tf.to_float((tf.math.equal(max_idx, cls_i) & (max_probs>thres[cls_i])))
                    pseudo_mask += tf.to_float(mask_cls_i)
                pseudo_mask =  tf.clip_by_value(pseudo_mask, clip_value_min=0.0, clip_value_max=1.0)                
                loss_xe_aug = tf.reduce_mean(loss_xe_aug * pseudo_mask)
                loss_xe = loss_xe + loss_xe_aug
                tf.summary.scalar('losses/xe_aug', loss_xe_aug)
                tf.summary.scalar('monitors/mask_aug', tf.reduce_mean(pseudo_mask))

        '''
        if FLAGS.db == 2:
            pseudo_labels_aug = tf.stop_gradient(tf.nn.softmax(logits_x))
            loss_xe_aug = weighted_ce_loss_aug(output_lb=logits_aug_strong, targets=tf.argmax(pseudo_labels_aug, axis=1), 
                output_ulb=logits_weak, p_cls_ema=p_in, upper=FLAGS.upper)
            thres = confidence * (update_p/tf.reduce_max(update_p))**0.5
            max_idx = tf.argmax(pseudo_labels_aug, axis=1)
            max_probs = tf.reduce_max(pseudo_labels_aug, axis=1)
            pseudo_mask = tf.zeros_like(max_probs)
            for cls_i in range(self.dataset.nclass):
                mask_cls_i = tf.to_float((tf.math.equal(max_idx, cls_i) & (max_probs>thres[cls_i])))
                pseudo_mask += tf.to_float(mask_cls_i)
            pseudo_mask =  tf.clip_by_value(pseudo_mask, clip_value_min=0.0, clip_value_max=1.0)                
            loss_xe_aug = tf.reduce_mean(loss_xe_aug * pseudo_mask)
            tf.summary.scalar('losses/xe_aug', loss_xe_aug)
            tf.summary.scalar('monitors/mask_aug', tf.reduce_mean(pseudo_mask))

            loss_xe = loss_xe + loss_xe_aug

        if FLAGS.db == 1:
            pseudo_labels_aug = tf.stop_gradient(tf.nn.softmax(logits_x))
            loss_xe_aug = weighted_ce_loss_aug(output_lb=logits_aug_strong, targets=tf.argmax(pseudo_labels_aug, axis=1), 
                output_ulb=logits_weak, p_cls_ema=p_in, upper=FLAGS.upper)
            thres = confidence * (update_p/tf.reduce_max(update_p))
            max_idx = tf.argmax(pseudo_labels_aug, axis=1)
            max_probs = tf.reduce_max(pseudo_labels_aug, axis=1)
            pseudo_mask = tf.zeros_like(max_probs)
            for cls_i in range(self.dataset.nclass):
                mask_cls_i = tf.to_float((tf.math.equal(max_idx, cls_i) & (max_probs>thres[cls_i])))
                pseudo_mask += tf.to_float(mask_cls_i)
            pseudo_mask =  tf.clip_by_value(pseudo_mask, clip_value_min=0.0, clip_value_max=1.0)                
            loss_xe_aug = tf.reduce_mean(loss_xe_aug * pseudo_mask)
            tf.summary.scalar('losses/xe_aug', loss_xe_aug)
            tf.summary.scalar('monitors/mask_aug', tf.reduce_mean(pseudo_mask))

            loss_xe = loss_xe + loss_xe_aug
        '''

        tf.summary.scalar('losses/xe', loss_xe)

        # Pseudo-label cross entropy for unlabeled data
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))

        if FLAGS.weight_l_ce == 3:
            loss_xeu, _ = weighted_ce_loss(output_lb=logits_strong, targets=tf.argmax(pseudo_labels, axis=1), 
                output_ulb=logits_weak, weight_l_ce=FLAGS.weight_l_ce, p_cls_ema=p_in, 
                step=self.step, upper=FLAGS.upper)
        else:
            loss_xeu = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(pseudo_labels, axis=1),
                                                                  logits=logits_strong)
        if FLAGS.weight_ulb == 0:
            pseudo_mask = tf.to_float(tf.reduce_max(pseudo_labels, axis=1) >= confidence)

        elif FLAGS.weight_ulb == 1:
            if FLAGS.weight_l_ce == 0:
                p_each_cls = get_p_cls(output_lb=logits_x, targets=l_in, output_ulb=logits_weak, weight_l_ce=FLAGS.weight_l_ce)
                update_p = tf.add(0.99*p_in, 0.01*p_each_cls)
            thres = confidence * (update_p/tf.reduce_max(update_p))**0.5
            max_idx = tf.argmax(pseudo_labels, axis=1)
            max_probs = tf.reduce_max(pseudo_labels, axis=1)
            pseudo_mask = tf.zeros_like(max_probs)
            for cls_i in range(self.dataset.nclass):
                mask_cls_i = tf.to_float((tf.math.equal(max_idx, cls_i) & (max_probs>thres[cls_i])))
                pseudo_mask += tf.to_float(mask_cls_i)
            pseudo_mask =  tf.clip_by_value(pseudo_mask, clip_value_min=0.0, clip_value_max=1.0)                

        elif FLAGS.weight_ulb == 2:
            if FLAGS.weight_l_ce == 0:
                p_each_cls = get_p_cls(output_lb=logits_x, targets=l_in, output_ulb=logits_weak, weight_l_ce=FLAGS.weight_l_ce)
                update_p = tf.add(0.99*p_in, 0.01*p_each_cls)
            thres = confidence * (update_p/tf.reduce_max(update_p))
            max_idx = tf.argmax(pseudo_labels, axis=1)
            max_probs = tf.reduce_max(pseudo_labels, axis=1)
            pseudo_mask = tf.zeros_like(max_probs)
            for cls_i in range(self.dataset.nclass):
                mask_cls_i = tf.to_float((tf.math.equal(max_idx, cls_i) & (max_probs>thres[cls_i])))
                pseudo_mask += tf.to_float(mask_cls_i)
            pseudo_mask =  tf.clip_by_value(pseudo_mask, clip_value_min=0.0, clip_value_max=1.0)                

        elif FLAGS.weight_ulb == 3:
            if FLAGS.weight_l_ce == 0:
                p_each_cls = get_p_cls(output_lb=logits_x, targets=l_in, output_ulb=logits_weak, weight_l_ce=FLAGS.weight_l_ce)
                update_p = tf.add(0.99*p_in, 0.01*p_each_cls)**0.2
            thres = confidence * (update_p/tf.reduce_max(update_p))
            max_idx = tf.argmax(pseudo_labels, axis=1)
            max_probs = tf.reduce_max(pseudo_labels, axis=1)
            pseudo_mask = tf.zeros_like(max_probs)
            for cls_i in range(self.dataset.nclass):
                mask_cls_i = tf.to_float((tf.math.equal(max_idx, cls_i) & (max_probs>thres[cls_i])))
                pseudo_mask += tf.to_float(mask_cls_i)
            pseudo_mask =  tf.clip_by_value(pseudo_mask, clip_value_min=0.0, clip_value_max=1.0)                

        elif FLAGS.weight_ulb == 4:
            if FLAGS.weight_l_ce == 0:
                p_each_cls = get_p_cls(output_lb=logits_x, targets=l_in, output_ulb=logits_weak, weight_l_ce=FLAGS.weight_l_ce)
                update_p = tf.add(0.99*p_in, 0.01*p_each_cls)**2
            thres = confidence * (update_p/tf.reduce_max(update_p))
            max_idx = tf.argmax(pseudo_labels, axis=1)
            max_probs = tf.reduce_max(pseudo_labels, axis=1)
            pseudo_mask = tf.zeros_like(max_probs)
            for cls_i in range(self.dataset.nclass):
                mask_cls_i = tf.to_float((tf.math.equal(max_idx, cls_i) & (max_probs>thres[cls_i])))
                pseudo_mask += tf.to_float(mask_cls_i)
            pseudo_mask =  tf.clip_by_value(pseudo_mask, clip_value_min=0.0, clip_value_max=1.0)                

        elif FLAGS.weight_ulb == 5:
            if FLAGS.weight_l_ce == 0:
                p_each_cls = get_p_cls(output_lb=logits_x, targets=l_in, output_ulb=logits_weak, weight_l_ce=FLAGS.weight_l_ce)
                update_p = tf.add(0.99*p_in, 0.01*p_each_cls)**0.2
            # update_p = 0
            e = np.arange(10*1.0) / (10 - 1.0)
            base = np.ones_like(e, np.float32) * 1.0/50
            not_update_p = np.float32(np.power(base, e))
            not_update_p = tf.convert_to_tensor(not_update_p)
            thres = confidence * (not_update_p/tf.reduce_max(not_update_p))
            max_idx = tf.argmax(pseudo_labels, axis=1)
            max_probs = tf.reduce_max(pseudo_labels, axis=1)
            pseudo_mask = tf.zeros_like(max_probs)
            for cls_i in range(self.dataset.nclass):
                mask_cls_i = tf.to_float((tf.math.equal(max_idx, cls_i) & (max_probs>thres[cls_i])))
                pseudo_mask += tf.to_float(mask_cls_i)
            pseudo_mask =  tf.clip_by_value(pseudo_mask, clip_value_min=0.0, clip_value_max=1.0)                

        # elif FLAGS.weight_ulb == 3:
        #     # dynamic threshold as DASH
        #     confidence = 0.0 
        #     confidence = tf.cond(self.step < 70000, lambda: 0.0, 
        #         lambda:tf.exp(- 1.0001* tf.stop_gradient(loss_xe) * 1.27**tf.to_float(70000-self.step)))  
        #     confidence = tf.cond(confidence>tf.exp(-0.05), lambda:tf.exp(-0.05), lambda:confidence)
        #     pseudo_mask = tf.to_float(tf.reduce_max(pseudo_labels, axis=1) >= confidence)
        #     tf.summary.scalar('monitors/confidence', confidence)

        tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))
        loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)
        tf.summary.scalar('losses/xeu', loss_xeu)

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        for i in range(self.dataset.nclass):
            tf.summary.scalar('losses/p_cls_'+str(i), update_p[i])

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss_xe + wu * loss_xeu + wd * loss_wd, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return utils.EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, p_cls_ema=p_in, train_op=train_op, update_p=update_p,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))


def log_softmax(x):
    xdev = x - tf.reduce_max(x, axis=1, keepdims=True)
    return xdev - tf.log(tf.reduce_sum(tf.exp(xdev), axis=1, keepdims=True))


def categorical_crossentropy_logdomain(log_predictions, targets):
    return -tf.reduce_sum(targets * log_predictions, axis=1)
 

def weighted_ce_loss(output_lb, targets, output_ulb, weight_l_ce, p_cls_ema=None, step=0, upper=10.0):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    probability_batch=tf.nn.softmax(output_ulb)
    p_each_cls = tf.reduce_mean(probability_batch, 0)
    p_each_cls_ori = tf.stop_gradient(p_each_cls) 
    if weight_l_ce >= 4:
        probability_batch_inv = tf.linalg.pinv(tf.stop_gradient(probability_batch))
        bs_weight = tf.matmul(tf.expand_dims(p_cls_ema,0), probability_batch_inv)
        bs_weight = tf.clip_by_value(bs_weight, clip_value_min=0.0, clip_value_max=upper)  
        p_each_cls = tf.matmul(bs_weight, probability_batch)
    target_one_hot = tf.zeros_like(output_lb)
    target_one_hot = tf.one_hot(indices=targets, depth=output_lb.get_shape()[1], on_value=1.0, off_value=0.0, axis=1, dtype=tf.float32)
    p_each_cls = tf.squeeze(p_each_cls)

    py = tf.reduce_sum(tf.multiply(target_one_hot, tf.expand_dims(p_each_cls, 0)), axis=-1) 
    px = tf.ones(tf.shape(targets)[0])
    n_class = tf.cast(tf.shape(targets)[0],  dtype=tf.float32)
    y_weight = n_class * (px/py)/tf.reduce_sum((px/py))
    
    y_weight = tf.expand_dims(y_weight, 1)

    if weight_l_ce == 2:
        y_weight = tf.stop_gradient(y_weight)
    log_P = log_softmax(output_lb) + tf.log(y_weight)
    ce_loss = categorical_crossentropy_logdomain(log_P, target_one_hot)
    return ce_loss, p_each_cls_ori
    # return ce_loss, tf.stop_gradient(p_each_cls) 


def get_p_cls(output_lb, targets, output_ulb, weight_l_ce):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """

    probability_batch=tf.nn.softmax(output_ulb)
    p_each_cls = tf.reduce_mean(probability_batch, 0)

    return tf.stop_gradient(p_each_cls) 


def weighted_ce_loss_aug(output_lb, targets, output_ulb, weight_l_ce, p_cls_ema=None, step=0, upper=10.0):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    probability_batch=tf.nn.softmax(output_ulb)
    p_each_cls = tf.reduce_mean(probability_batch, 0)
    p_each_cls_ori = tf.stop_gradient(p_each_cls) 
    if weight_l_ce >= 4:
        probability_batch_inv = tf.linalg.pinv(tf.stop_gradient(probability_batch))
        bs_weight = tf.matmul(tf.expand_dims(p_cls_ema,0), probability_batch_inv)
        bs_weight = tf.clip_by_value(bs_weight, clip_value_min=0.0, clip_value_max=upper)  
        p_each_cls = tf.matmul(bs_weight, probability_batch)
    target_one_hot = tf.zeros_like(output_lb)
    target_one_hot = tf.one_hot(indices=targets, depth=output_lb.get_shape()[1], on_value=1.0, off_value=0.0, axis=1, dtype=tf.float32)
    p_each_cls = tf.squeeze(p_each_cls)

    py = tf.reduce_sum(tf.multiply(target_one_hot, tf.expand_dims(p_each_cls, 0)), axis=-1) 
    px = tf.ones(tf.shape(targets)[0])
    n_class = tf.cast(tf.shape(targets)[0],  dtype=tf.float32)
    y_weight = n_class * (px/py)/tf.reduce_sum((px/py))
    y_weight = tf.expand_dims(y_weight, 1)
    log_P = log_softmax(output_lb) + tf.log(y_weight)
    ce_loss = categorical_crossentropy_logdomain(log_P, target_one_hot)
    
    # CE_LOSS_AUG
    pseudo_labels_aug = tf.stop_gradient(tf.nn.softmax(output_lb))
    target_one_hot_aug = tf.zeros_like(output_lb)
    target_one_hot_aug = tf.one_hot(indices=tf.argmax(pseudo_labels_aug, axis=1), depth=output_lb.get_shape()[1], on_value=1.0, off_value=0.0, axis=1, dtype=tf.float32)
    py_aug = tf.reduce_sum(tf.multiply(target_one_hot_aug, tf.expand_dims(p_each_cls, 0)), axis=-1) 
    y_weight_aug = n_class * (px/py_aug)/tf.reduce_sum((px/py_aug))
    y_weight_aug = tf.expand_dims(y_weight_aug, 1)
    log_P_aug =  tf.log(y_weight_aug)
    ce_loss_aug = categorical_crossentropy_logdomain(log_P_aug, target_one_hot_aug)

    return ce_loss, -ce_loss_aug, p_each_cls_ori
    # return ce_loss_ori


def reweight_ce_loss(output_lb, targets):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    nclass = 10
    e = np.arange(nclass*1.0) / (nclass - 1.0)
    base = np.ones_like(e, np.float32) * 1.0/50
    gt_p_data = np.power(base, e)

    target_one_hot = tf.zeros_like(output_lb)
    target_one_hot = tf.one_hot(indices=targets, depth=output_lb.get_shape()[1], on_value=1.0, off_value=0.0, axis=1, dtype=tf.float32)

    py = tf.reduce_sum(tf.multiply(target_one_hot, tf.cast(tf.expand_dims(gt_p_data, 0), dtype=tf.float32)), axis=-1) 
    px = tf.ones(tf.shape(targets)[0])
    y_weight = (px/py)
    
    y_weight = tf.expand_dims(y_weight, 1)

    log_P = log_softmax(output_lb) * y_weight
    ce_loss = categorical_crossentropy_logdomain(log_P, target_one_hot)
    return ce_loss
    # return ce_loss, tf.stop_gradient(p_each_cls) 


def la_ce_loss(output_lb, targets):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    nclass = 10
    e = np.arange(nclass*1.0) / (nclass - 1.0)
    base = np.ones_like(e, np.float32) * 1.0/50
    gt_p_data = np.power(base, e)
    gt_p_data = gt_p_data/sum(gt_p_data)

    target_one_hot = tf.zeros_like(output_lb)
    target_one_hot = tf.one_hot(indices=targets, depth=output_lb.get_shape()[1], on_value=1.0, off_value=0.0, axis=1, dtype=tf.float32)

    py = tf.reduce_sum(tf.multiply(target_one_hot, tf.cast(tf.expand_dims(gt_p_data, 0), dtype=tf.float32)), axis=-1) 
    
    y_weight = tf.expand_dims(py, 1)

    log_P = log_softmax(output_lb + tf.log(y_weight)) 
    ce_loss = categorical_crossentropy_logdomain(log_P, target_one_hot)
    return ce_loss


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = FixMatch(
        os.path.join(FLAGS.train_dir, dataset.name, FixMatch.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        wu=FLAGS.wu,
        confidence=FLAGS.confidence,
        uratio=FLAGS.uratio,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat,
        weight_l_ce=FLAGS.weight_l_ce,
        devicenum=FLAGS.devicenum,
        weight_ulb=FLAGS.weight_ulb,
        # use_bn=FLAGS.use_bn,
        train_kimg=FLAGS.train_kimg,
        upper=FLAGS.upper,
        db=FLAGS.db,
        )

    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Pseudo label loss weight.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('uratio', 7, 'Unlabeled batch size ratio.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 13)
    # FLAGS.set_default('train_kimg', 1 << 16)
    
    flags.DEFINE_integer('weight_l_ce', 0, 'weighted ce loss.')
    flags.DEFINE_integer('devicenum', 1, 'number of device')
    flags.DEFINE_integer('weight_ulb', 0, 'weighted threshold for unlabeled data')
    flags.DEFINE_float('upper', 10.0, 'upper for the matrix inversion')

    flags.DEFINE_integer('db', 0, 'doubly robust.')

    # flags.DEFINE_integer('use_bn', 1, 'use bn or not--for debugging')

    app.run(main)
