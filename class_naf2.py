import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import importlib, pkg_resources
import tensorflow as tf

from tensorflow.keras import layers
import tensorflow_probability as tfp
import gym, cirq, sympy
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from collections import deque, namedtuple ,defaultdict
import copy
import random
import pickle


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


        self.n_step_buffer = deque(maxlen=1)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = tf.convert_to_tensor(np.stack([e.state for e in experiences if e is not None]),dtype=tf.float32)
        actions = tf.convert_to_tensor(np.vstack([e.action for e in experiences if e is not None]),dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.vstack([e.reward for e in experiences if e is not None]),dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.stack([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
        dones = tf.convert_to_tensor(np.vstack([e.done for e in experiences if e is not None]),dtype=tf.float32)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class NAF(tf.keras.Model):
  def __init__(self,n_layers,n_actions):
    super(NAF,self).__init__()
    self.n_actions=n_actions
    self.fc1=layers.Dense(n_layers,activation='relu',kernel_regularizer='l2')
    self.fc2=layers.Dense(n_layers,activation='relu',kernel_regularizer='l2')
    self.fc3=layers.Dense(n_layers,activation='relu',kernel_regularizer='l2')
    self.fc4=layers.Dense(n_layers,activation='relu',kernel_regularizer='l2')
    self.value=layers.Dense(1, activation=None,kernel_regularizer='l2')
    self.action_values=layers.Dense(self.n_actions,activation=None,kernel_regularizer='l2')
    self.diag_matrix_entries=layers.Dense(int(self.n_actions),activation=None,kernel_regularizer='l2')
    
  def call(self,inputs,action=None,take_action=True):
    x=self.fc1(inputs)
    x=self.fc2(x)
    x=self.fc3(x)
    x=self.fc4(x)

    action_value=2*tf.math.tanh(self.action_values(x))
    # low_entries = tf.math.tanh(self.low_matrix_entries(x))
    diga_entries = tf.math.exp(tf.math.tanh(self.diag_matrix_entries(x)))
    V=self.value(x)
    
    action_value=tf.expand_dims(action_value,axis=-1)  

    L = tf.zeros((inputs.shape[0],self.n_actions,self.n_actions),dtype=V.dtype)
    # get lower triagular and diganal indices
    # tril_indic=tf.transpose(tf.constant(np.tril_indices(n=self.n_actions,m=self.n_actions,k=-1)))
    dig_ind= tf.transpose( tf.tile( tf.constant(np.arange(self.n_actions),dtype=tf.int32,shape=(1,self.n_actions)) ,(2,1) )  )
    # t_ind= tf.concat([tril_indic,dig_ind],0)
    # entris= tf.concat([low_entries,diga_entries],1)

    L=tf.transpose(L,perm=(1,2,0))
    # L=tf.tensor_scatter_nd_update(L,t_ind,tf.transpose(entris))
    L=tf.tensor_scatter_nd_update(L,dig_ind,tf.transpose(diga_entries))
    L=tf.transpose(L,perm=(2,0,1))

    # calculate state-dependent, positive-definite square matrix
    P=L@tf.transpose(L,perm=(0,2,1))

    Q=None
    
    if action is not None:
      #calculate Advantage:
      A =tf.squeeze(-0.5 * tf.matmul( tf.matmul(tf.transpose(tf.expand_dims(action,axis=-1) - action_value, perm= (0,2,1)), P) , 
                            tf.expand_dims(action,axis=-1) - action_value ), axis=-1 )
      Q= A+V
    if take_action==True:
        dist= tfp.distributions.MultivariateNormalLinearOperator(loc=tf.squeeze(action_value,axis=-1), scale=tf.linalg.LinearOperatorLowerTriangular(tf.linalg.inv(L)))
        action = dist.sample()
        action = tf.clip_by_value(action,-2,2)
    action_value=tf.squeeze(action_value,axis=-1)
    return action,Q,V, action_value

class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 NUPDATES,
                 ):


        self.action_size = action_size
        # self.seed = random.seed(seed)
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.NUPDATES = NUPDATES
        self.BATCH_SIZE = BATCH_SIZE
        

        # Q-Network
        self.qnetwork_local = NAF(layer_size, action_size )#qubits,n_layers,n_actions,observables 
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)

        self.optimizer =tf.keras.optimizers.Adam(learning_rate=LR, amsgrad=True)
        # self.optimizer2 =tf.keras.optimizers.Adam(learning_rate=LR2, amsgrad=True)

        print(self.qnetwork_local)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Noise process
        # self.noise = OUNoise(action_size)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                for _ in range(self.NUPDATES):
                    experiences = self.memory.sample()
                    loss = self.learn(experiences)
                  
          
    def act(self, state):
        """Calculating the action
        Params
        ======
            state (array_like): current state  
        """

        state = tf.expand_dims(tf.convert_to_tensor((state),dtype=tf.float32),axis=0  )  
        action, _, _,_ = self.qnetwork_local(state)
        return tf.squeeze(action,axis=0).numpy()
    
    def act_without_noise(self, state):
        state = tf.expand_dims(tf.convert_to_tensor((state),dtype=tf.float32),axis=0  )  
        _, _, _,action = self.qnetwork_local(state)
        return tf.squeeze(action,axis=0).numpy()


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # get the Value for the next state from target model
   
        _, _, V_,_ = self.qnetwork_target(next_states,take_action=False)

        # Compute Q targets for current states 
        V_targets = rewards + (self.GAMMA * V_ * (1 - dones))
        
        with tf.GradientTape() as tape:
        # Get expected Q values from local model
          tape.watch(self.qnetwork_local.trainable_variables)
          _,Q, _ ,_= self.qnetwork_local(states, actions,take_action=False)
          loss = tf.keras.losses.MeanSquaredError()(V_targets, Q)
      
        grads = tape.gradient(loss,self.qnetwork_local.trainable_variables)

        self.optimizer.apply_gradients(zip(grads,self.qnetwork_local.trainable_variables))
        # self.optimizer.apply_gradients([(grads[0],self.qnetwork_local.trainable_variables[0])])
        # self.optimizer2.apply_gradients(zip(grads[1:],self.qnetwork_local.trainable_variables[1:]))
        # ------------------- update target network ------------------- #
        self.soft_update()
        
        # self.noise.reset()
        #self.noise.reset()
        return loss.numpy() 



    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        # self.qnetwork_local.set_weights([self.TAU*i+(1.0-self.TAU)*j for i,j in zip(self.qnetwork_local.get_weights(),self.qnetwork_target.get_weights())])
        self.qnetwork_target.set_weights([self.TAU*i+(1.0-self.TAU)*j for i,j in zip(self.qnetwork_local.get_weights(),self.qnetwork_target.get_weights())])

def evaluate(eval_runs):
    scores2 = []
    # test_env=gym.make('Pendulum-v0', g=9.81)
    for i in range(eval_runs):
        state = test_env.reset()    
        score = 0
        done = 0
        while not done:
            action = agent.act_without_noise(state)
            state, reward, done, _ = test_env.step(action)
            score += reward
            if done:
                scores2.append(score)
                break
    return np.mean(scores2)



BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.9
TAU = 0.01
LR = 1e-2
# LR2=1e-2
UPDATE_EVERY = 4
NUPDATES = 1
env=gym.make('Pendulum-v0', g=9.81)
#env=gym.make('Pendulum-v0', g=9.81)
test_env=gym.make('Pendulum-v0', g=9.81)


action_size = 1

for i in range(10):
    agent = DQN_Agent(  
                        action_size=action_size,
                        layer_size=15,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA, 
                        UPDATE_EVERY=UPDATE_EVERY,
                        NUPDATES=NUPDATES)

    # frames=1000000
    scores = []  
    evas=[]                      # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    i_episode = 0
    state = env.reset()
    step=0
    score = 0  

    # agent.qnetwork_local.load_weights('./QUANTUM')
    # agent.qnetwork_target.load_weights('./QUANTUM')
    #1001
    while i_episode <1001:
        action = agent.act(state)
        step+=1
        next_state, reward, done, _ = env.step(np.array([action]))
        next_state= np.reshape(next_state,3)# /np.array([1.0,1.0,8.0])
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        
        
            
        if done:
            # scores_window.append(score)       # save most recent score
            # scores.append(score)              # save most recent score
            print('episode=',i_episode)

            if i_episode % 5 == 0:       
                eva=evaluate(eval_runs=3)
                evas.append(eva)
                print('evaluate=',eva)
    
            if i_episode % 100 == 0:  
                # with open('./Class/evas20_nobn','wb') as f:  
                ##evas20 is actually layer size 15, i just named wrong
                with open(f'./Class_l4_20/evas20_nobn_{i+20}','wb') as f:
                    pickle.dump(evas,f)
            i_episode +=1 
            state = env.reset()
            # state=state/np.array([1.0,1.0,8.0])
            step=0
            score = 0  
    with open(f'./Class_l4_20/evas20_nobn_{i+20}','wb') as f:
        pickle.dump(evas,f)