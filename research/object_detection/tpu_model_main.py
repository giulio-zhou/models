# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Creates and runs `Estimator` for object detection model on TPUs.

This uses the TPUEstimator API to define and run a model in TRAIN/EVAL modes.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

from object_detection import model_hparams
from object_detection import model_lib

import subprocess

tf.flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than plain CPUs')

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers.')

flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU cores).')
flags.DEFINE_integer('iterations_per_loop', 100,
                     'Number of iterations per TPU training loop.')
# For mode=train_and_eval, evaluation occurs after training is finished.
# Note: independently of steps_per_checkpoint, estimator will save the most
# recent checkpoint every 10 minutes by default for train_and_eval
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train, eval, train_and_eval')
flags.DEFINE_integer('train_batch_size', None, 'Batch size for training. If '
                     'this is not provided, batch size is read from training '
                     'config.')

flags.DEFINE_string(
    'hparams_overrides', None, 'Comma-separated list of '
    'hyperparameters to override defaults.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_integer('eval_every_n_iters', None, 'Number of training steps '
                     'between evals. Must be divisible by 100 and must be '
                     'provided in train_and_eval mode.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
    'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')

FLAGS = tf.flags.FLAGS

def load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = tf.train.NewCheckpointReader(
        tf.train.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0

def main(unused_argv):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')

  tpu_cluster_resolver = (
      tf.contrib.cluster_resolver.TPUClusterResolver(
          tpu=FLAGS.tpu_name.split(','),
          zone=FLAGS.tpu_zone,
          project=FLAGS.gcp_project))
  tpu_grpc_url = tpu_cluster_resolver.get_master()

  config = tf.contrib.tpu.RunConfig(
      master=tpu_grpc_url,
      evaluation_master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_shards),
      save_checkpoints_steps=FLAGS.eval_every_n_iters)

  kwargs = {}
  if FLAGS.train_batch_size:
    kwargs['batch_size'] = FLAGS.train_batch_size

  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          FLAGS.sample_1_of_n_eval_on_train_examples),
      use_tpu_estimator=True,
      use_tpu=FLAGS.use_tpu,
      num_shards=FLAGS.num_shards,
      **kwargs)
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fns = train_and_eval_dict['eval_input_fns']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  train_steps = train_and_eval_dict['train_steps']

  if FLAGS.mode == 'train':
    estimator.train(input_fn=train_input_fn, max_steps=train_steps)

  # Continuously evaluating.
  if FLAGS.mode == 'eval':
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      # Currently only a single eval input is allowed.
      input_fn = eval_input_fns[0]
    model_lib.continuous_eval(estimator, FLAGS.model_dir, input_fn, train_steps,
                              name)

  if FLAGS.mode == 'train_and_eval':
    # load last checkpoint and start from there
    current_step = load_global_step_from_checkpoint_dir(FLAGS.model_dir)
    
    while current_step < train_steps:
      # Train for up to steps_per_eval number of steps.
      # At the end of training, a checkpoint will be written to --model_dir.
      next_checkpoint = min(current_step + FLAGS.eval_every_n_iters, train_steps)
      estimator.train(input_fn=train_input_fn, max_steps=next_checkpoint)
      current_step = next_checkpoint
      # Evaluate the model on the most recent model in --model_dir.
      # Since evaluation happens in batches of --eval_batch_size, some images
      # may be excluded modulo the batch size. As long as the batch size is
      # consistent, the evaluated images are also consistent.
      tf.logging.info('Starting to evaluate at step %d', next_checkpoint)
      # Background evaluation process.
      args = ['PYTHONPATH=$PYTHONPATH:slim',
              'python', 'object_detection/model_main.py',
              '--pipeline_config_path', FLAGS.pipeline_config_path,
              '--model_dir', FLAGS.model_dir,
              '--checkpoint_dir', FLAGS.model_dir,
              '--sample_1_of_n_eval_examples', str(FLAGS.sample_1_of_n_eval_examples),
              '--run_once', str(True)]
      print(' '.join(args))
      p = subprocess.Popen(' '.join(args), stderr=subprocess.STDOUT, shell=True)



if __name__ == '__main__':
  tf.app.run()
