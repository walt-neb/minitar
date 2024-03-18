# minitour training script, using the SAC Reinforced Learning method.
# This work is based on the tutorial provided by the tf-agents example for minitour
# add adds functionality to help organize and log training exersises.

import helper_functions as hf
import os
import sys
import numpy as np
import tensorflow as tf
import PIL.Image
import datetime
import reverb
import base64
import imageio
import IPython
import matplotlib.pyplot as plt

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network

# Suppress PyBullet output
sys.stdout = open(os.devnull, 'w')
from tf_agents.environments import suite_pybullet
# Reset stdout to its original state
sys.stdout = sys.__stdout__

from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy, py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils

os.environ['TF_USE_LEGACY_KERAS'] = '1'
tempdir = '../temp/'  # Modify as needed
checkpoint_dir = '../temp/policies/checkpoints'  # or construct this path dynamically as needed
#restore_checkpoint = 'policy_checkpoint_0000119000/'#'/policy_checkpoint_0000095000/' #'policy_checkpoint_0000019000/'

# Hyperparameter Configuration Start
# Call this function to update the hyperparameters
hyp = hf.read_hyperparameters()# Hyperparameter Configuration Start

# Call this function at the beginning of your script to update the hyperparameters
#read_hyperparameters()

env_name = "MinitaurBulletEnv-v0"  # @param {type:"string"}
env = suite_pybullet.load(env_name)
env.reset()
PIL.Image.fromarray(env.render())
now = datetime.datetime.now()
print("Starting at {}".format(now))

collect_env = suite_pybullet.load(env_name)
eval_env = suite_pybullet.load(env_name)
print('colleting_env = '.format(collect_env))
print('eval_env = '.format(eval_env))


use_gpu = True
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_env)

with strategy.scope():
  critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=hyp['critic_joint_fc_layer_params'],
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

  with strategy.scope():
      actor_net = actor_distribution_network.ActorDistributionNetwork(
          observation_spec,
          action_spec,
          fc_layer_params=hyp['actor_fc_layer_params'],
          continuous_projection_net=(
              tanh_normal_projection_network.TanhNormalProjectionNetwork))

with strategy.scope():
  train_step = train_utils.create_train_step()

  tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=hyp['actor_learning_rate']),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=hyp['critic_learning_rate']),
        alpha_optimizer=tf.keras.optimizers.Adam(
            learning_rate=hyp['alpha_learning_rate']),
        target_update_tau=hyp['target_update_tau'],
        target_update_period=hyp['target_update_period'],
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=hyp['gamma'],
        reward_scale_factor=hyp['reward_scale_factor'],
        train_step_counter=train_step)

  tf_agent.initialize()

    # Rest of the script remains as in your original script, with hyperparameter variables replaced as needed.

    # When initializing agent_learner, check if checkpoint_path is not None, then load the checkpoint


'''
Replay Buffer
In order to keep track of the data collected from the environment, we will use Reverb, an efficient, extensible, 
and easy-to-use replay system by Deepmind. It stores experience data collected by the Actors and consumed by the 
Learner during training.
In this tutorial, this is less important than max_size -- but in a distributed setting with async collection and 
training, you will probably want to experiment with rate_limiters.SampleToInsertRatio, using a samples_per_insert 
somewhere between 2 and 1000. For example:
rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)
'''

table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=hyp['replay_buffer_capacity'],
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])

reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)

dataset = reverb_replay.as_dataset(
      sample_batch_size=hyp['batch_size'], num_steps=2).prefetch(50)
experience_dataset_fn = lambda: dataset

'''
Policies
In TF-Agents, policies represent the standard notion of policies in RL: given a time_step produce an action or a distribution over actions. The main method is policy_step = policy.step(time_step) where policy_step is a named tuple PolicyStep(action, state, info). The policy_step.action is the action to be applied to the environment, 
state represents the state for stateful (RNN) policies and info may contain auxiliary information such as log probabilities of the actions.
Agents contain two policies:

    agent.policy — The main policy that is used for evaluation and deployment.
    agent.collect_policy — A second policy that is used for data collection.
'''
tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_eval_policy, use_tf_function=True)

tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_collect_policy, use_tf_function=True)

#Policies can be created independently of agents. For example, use tf_agents.policies.random_py_policy to create
# a policy which will randomly select an action for each time_step.

random_policy = random_py_policy.RandomPyPolicy(
  collect_env.time_step_spec(), collect_env.action_spec())


'''
Actors
The actor manages interactions between a policy and an environment.

    The Actor components contain an instance of the environment (as py_environment) and a copy of the policy variables.
    Each Actor worker runs a sequence of data collection steps given the local values of the policy variables.
    Variable updates are done explicitly using the variable container client instance in the training script before calling actor.run().
    The observed experience is written into the replay buffer in each data collection step.

As the Actors run data collection steps, they pass trajectories of (state, action, reward) to the observer, which caches and writes them to the Reverb replay system.

We're storing trajectories for frames [(t0,t1) (t1,t2) (t2,t3), ...] because stride_length=1.
'''
rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  reverb_replay.py_client,
  table_name,
  sequence_length=2,
  stride_length=1)

initial_collect_actor = actor.Actor(
  collect_env,
  random_policy,
  train_step,
  steps_per_run=hyp['initial_collect_steps'],
  observers=[rb_observer])
initial_collect_actor.run()

env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
  collect_env,
  collect_policy,
  train_step,
  steps_per_run=1,
  metrics=actor.collect_metrics(10),
  summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
  observers=[rb_observer, env_step_metric])

eval_actor = actor.Actor(
  eval_env,
  eval_policy,
  train_step,
  episodes_per_run=hyp['num_eval_episodes'],
  metrics=actor.eval_metrics(hyp['num_eval_episodes']),
  summary_dir=os.path.join(tempdir, 'eval'),
)

saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)
print('learner.POLICY_SAVED_MODEL_DIR = ', saved_model_dir)

if sys.argv[1] is not None:
    my_checkpoint = sys.argv[1]
    checkpoint_to_load = os.path.join(checkpoint_dir, my_checkpoint, 'variables', 'variables')
    #checkpoint_to_load = os.path.join(checkpoint_dir, 'policy_checkpoint_0000195000', 'variables', 'variables')
    print(f"Loading policy checkpoint: {checkpoint_to_load}")
    checkpoint = tf.train.Checkpoint(agent=tf_agent)
    status = checkpoint.restore(checkpoint_to_load)

# ----------- use reverb checkpoints -------------------
# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        interval=hyp['policy_save_interval']),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]

agent_learner = learner.Learner(
  tempdir,
  train_step,
  tf_agent,
  experience_dataset_fn,
  triggers=learning_triggers,
  strategy=strategy)


def get_eval_metrics():
  eval_actor.run()
  results = {}
  for metric in eval_actor.metrics:
    results[metric.name] = metric.result()
  return results

metrics = get_eval_metrics()

def log_eval_metrics(step, metrics):
  eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
  print('step = {0}: {1}'.format(step, eval_results))

log_eval_metrics(0, metrics)

print("Training the agent...")
'''
=====================================================================================  
Training the agent¶ 
The training loop involves both collecting data from the environment and optimizing 
the agent's networks. Along the way, we will occasionally evaluate the agent's policy 
to see how we are doing.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
=====================================================================================
'''
# Reset the train step

t_start = datetime.datetime.now()
#display time in H:M:S
print("Starting at {}".format(t_start))
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]
print("Started training with the following returns: ")
print(returns)

for _ in range(hyp['num_iterations']):
  # ============== Training ========================================================================
  collect_actor.run()
  loss_info = agent_learner.run(iterations=1)

  # Evaluating.
  step = agent_learner.train_step_numpy

  if hyp['eval_interval'] and step % hyp['eval_interval'] == 0:
    metrics = get_eval_metrics()
    log_eval_metrics(step, metrics)
    returns.append(metrics["AverageReturn"])

  if hyp['log_interval'] and step % hyp['log_interval'] == 0:
    print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

rb_observer.close()
reverb_server.stop()

# ------------------ store the hyperparameters used for this training session ------

t_end = datetime.datetime.now()
#display time in H:M:S
print("==============================================")
print("Started  Training at {}".format(t_start))
print("Finished Training at {}".format(t_end))
print("Total time taken {}".format(t_end - t_start))
print("==============================================")

hf.log_training_and_hyperparameters(hyp,t_start,t_end)

'''
Plots¶
We can plot average return vs global steps to see the performance of our agent. 
In Minitaur, the reward function is based on how far the minitaur walks in 1000 
steps and penalizes the energy expenditure.
'''
steps = range(0, hyp['num_iterations'] + 1, hyp['eval_interval'])
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()


'''
It is helpful to visualize the performance of an agent by rendering the environment at each step. 
Before we do that, let us first create a function to embed videos in this colab.
'''
def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

# Before ending the script, explicitly disconnect the PyBullet environment
if 'env' in locals():
    env.close()
if 'eval_env' in locals():
    eval_env.close()

print("creating video to demonstrate model performance...")
# generate random integer between 1 and 100
x = str(round(np.random.rand(), 2))
y = str(hyp['num_iterations']/1000)
num_episodes = 3
video_filename = './vids/sac_minitaur_' + x + '-' + y + 'k' + '.mp4'

print('Writing video {}'.format(video_filename))
with imageio.get_writer(video_filename, fps=50) as video:
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        video.append_data(eval_env.render())
        while not time_step.is_last():
            action_step = eval_actor.policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            video.append_data(eval_env.render())

embed_mp4(video_filename)

# Explicitly close the video writer and any other open resources
video.close()
print("Video has been created and saved.")

# End the script gracefully
print("Ending the script gracefully.")
exit(0)
