# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import tensorflow as tf
#import tf_agents
import base64
import imageio
import IPython
import matplotlib.pyplot as plt
import os
import reverb
import tempfile
import PIL.Image

import datetime


from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

tempdir = '/home/walt/src/temp/' #tempfile.gettempdir()


#=========================================================================
# ========================== hypterparameters ==========================
env_name = "MinitaurBulletEnv-v0"  # @param {type:"string"}
# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations              = 50000  # @param {type:"integer"}
initial_collect_steps       = 5000  # @param {type:"integer"}
collect_steps_per_iteration = 4  # @param {type:"integer"}
replay_buffer_capacity      = 50000  # ll
# @param {type:"integer"}
batch_size                  = 256  # @param {type:"integer"}
critic_learning_rate        = 3e-4  # @param {type:"number"}
actor_learning_rate         = 3e-4  # @param {type:"number"}
alpha_learning_rate         = 3e-4  # @param {type:"number"}
target_update_tau           = 0.005  # @param {type:"number"}
target_update_period        = 1  # @param {type:"number"}
gamma                       = 0.99  # @param {type:"number"}
reward_scale_factor         = 1.0  # @param {type:"number"}
actor_fc_layer_params           = (256, 256)
critic_joint_fc_layer_params    = (256, 256)
log_interval                = 5000  # @param {type:"integer"}
num_eval_episodes           = 200  # @param {type:"integer"}
eval_interval               = 10000  # @param {type:"integer"}
policy_save_interval        = 5000  # @param {type:"integer"}
#================================================================================
# =========================== load environement =================================
env = suite_pybullet.load(env_name)
env.reset()
PIL.Image.fromarray(env.render())
now = datetime.datetime.now()
#display time in H:M:S
print("Starting at {}".format(now))


if(False):
    import pybullet as p
    #urdf_file = "/home/walt/src/sac/venv/lib/python3.10/site-packages/pybullet_data/quadruped/minitaur.urdf"
    urdf_file = "/home/walt/src/sac/venv/lib/python3.10/site-packages/pybullet_data/quadruped/minitaur_v1.urdf"
    robot = p.loadURDF(urdf_file)
    for joint_index in range(p.getNumJoints(robot)):
        print(p.getJointInfo(robot, joint_index))
    #foot_index = # The index of the foot link
    #state = p.getLinkState(robot, foot_index)
    #position = state[0]  # Position of the foot link in Cartesian coordinates (x, y, z)


if(False):
    print('Observation Spec:')
    print(env.time_step_spec().observation)
    print('Action Spec:')
    print(env.action_spec())


collect_env = suite_pybullet.load(env_name)
eval_env = suite_pybullet.load(env_name)
print(collect_env)
print(eval_env)

use_gpu = True #@param {type:"boolean"}

strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)


observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(collect_env))

with strategy.scope():
  critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

  with strategy.scope():
      actor_net = actor_distribution_network.ActorDistributionNetwork(
          observation_spec,
          action_spec,
          fc_layer_params=actor_fc_layer_params,
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
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step)

  tf_agent.initialize()

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
    max_size=replay_buffer_capacity,
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
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
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
  steps_per_run=initial_collect_steps,
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
  episodes_per_run=num_eval_episodes,
  metrics=actor.eval_metrics(num_eval_episodes),
  summary_dir=os.path.join(tempdir, 'eval'),
)

saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)


# ----------- use reverb checkpoints -------------------

if(True):
    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
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
print("training starting with: ")
print(returns)

for _ in range(num_iterations):
  # Training.
  collect_actor.run()
  loss_info = agent_learner.run(iterations=1)

  # Evaluating.
  step = agent_learner.train_step_numpy

  if eval_interval and step % eval_interval == 0:
    metrics = get_eval_metrics()
    log_eval_metrics(step, metrics)
    returns.append(metrics["AverageReturn"])

  if log_interval and step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

rb_observer.close()
reverb_server.stop()

t_end = datetime.datetime.now()
#display time in H:M:S
print("==============================================")
print("Started  Training at {}".format(t_start))
print("Finished Training at {}".format(t_end))
print("Total time taken {}".format(t_end - t_start))
print("==============================================")

'''
Plots¶
We can plot average return vs global steps to see the performance of our agent. 
In Minitaur, the reward function is based on how far the minitaur walks in 1000 
steps and penalizes the energy expenditure.
'''
steps = range(0, num_iterations + 1, eval_interval)
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

print("creating video to demonstrate model performance...")
# generate random integer between 1 and 100
x = str(round(np.random.rand(), 2))
y = str(num_iterations/1000)
num_episodes = 3
video_filename = './vids/sac_minitaur_'+x+'-'+y+'k'+'.mp4'

print('Writing video {}'.format(video_filename))
with imageio.get_writer(video_filename, fps=60) as video:
  for _ in range(num_episodes):
    time_step = eval_env.reset()
    video.append_data(eval_env.render())
    while not time_step.is_last():
      action_step = eval_actor.policy.action(time_step)
      time_step   = eval_env.step(action_step.action)
      video.append_data(eval_env.render())

embed_mp4(video_filename)



