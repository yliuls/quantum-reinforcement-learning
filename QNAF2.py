
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import tensorflow_quantum as tfq
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

# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def generate_circuit(qubits, n_layers):

    # Number of qubits
    n_qubits = len(qubits)
    
    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))
    
    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_qubits})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))
    
    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))
    
    return circuit, list(params.flat), list(inputs.flat)

class ReUploadingPQC(tf.keras.layers.Layer):

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )
        # self.theta = tf.Variable(
        #     initial_value=tf.zeros(shape=(1, len(theta_symbols)), dtype="float32"),
        #     trainable=True, name="thetas"
        # )
        
        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )
        
        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        
        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)        

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        # batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        # tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        tiled_up_inputs = tf.tile(inputs, multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        
        return self.computation_layer([tiled_up_circuits, joined_vars])

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

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
  def __init__(self,qubits,n_layers,n_actions,observables):
    super(NAF,self).__init__()
    self.n_actions=n_actions
    self.re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')
    # self.re_uploading_pqc2 = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')
    # self.re_uploading_pqc3 = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')
    self.value=layers.Dense(1, activation=None,kernel_regularizer='l2')
    self.action_values=layers.Dense(self.n_actions,activation=None,kernel_regularizer='l2')
    # self.low_matrix_entries=layers.Dense(int(self.n_actions*(self.n_actions-1)/2),activation=None)
    self.diag_matrix_entries=layers.Dense(int(self.n_actions),activation=None,kernel_regularizer='l2')
    
  def call(self,inputs,action=None,take_action=True):
    x=self.re_uploading_pqc(inputs)
    # x2=self.re_uploading_pqc2(inputs)
    # x3=self.re_uploading_pqc3(inputs)
    # x=tf.concat([x1,x2,x3],1)
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
                 observables,
                 qubits,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 LR2,
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
        self.qnetwork_local = NAF(qubits,layer_size, action_size,observables )#qubits,n_layers,n_actions,observables 
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)

        self.optimizer =tf.keras.optimizers.Adam(learning_rate=LR, amsgrad=True)
        self.optimizer2 =tf.keras.optimizers.Adam(learning_rate=LR2, amsgrad=True)

        print(self.qnetwork_local)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Noise process
        self.noise = OUNoise(action_size)
    
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
       
        self.optimizer.apply_gradients([(grads[0],self.qnetwork_local.trainable_variables[0])])
        self.optimizer2.apply_gradients(zip(grads[1:],self.qnetwork_local.trainable_variables[1:]))
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

n_qubits = 3# Dimension of the state vectors in CartPole
n_layers = 10 # Number of layers in the PQC


qubits = cirq.GridQubit.rect(1, n_qubits)
ops = [cirq.Z(q) for q in qubits]
observables1 = reduce((lambda x, y: x * y), ops) # Z_0*Z_1*Z_2*Z_3

ops2 = [cirq.X(q) for q in qubits]
observables2 = reduce((lambda x, y: x * y), ops2) # Z_0*Z_1*Z_2*Z_3

ops3 = [cirq.Y(q) for q in qubits]
observables3 = reduce((lambda x, y: x * y), ops3) # Z_0*Z_1*Z_2*Z_3

# observables= [observables1,observables2,observables3]
observables= [ops[0] ,ops[1],ops[2]]
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.9
TAU = 0.01
LR = 1e-3
LR2=1e-2
UPDATE_EVERY = 4
NUPDATES = 1
env=gym.make('Pendulum-v0', g=9.81)
#env=gym.make('Pendulum-v0', g=9.81)
test_env=gym.make('Pendulum-v0', g=9.81)


action_size = 1
for i in range(3):
    agent = DQN_Agent(observables,qubits,    
                        action_size=action_size,
                        layer_size=10,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        LR2=LR2,
                        TAU=TAU, 
                        GAMMA=GAMMA, 
                        UPDATE_EVERY=UPDATE_EVERY,
                        NUPDATES=NUPDATES)

    # frames=1000000
    scores = []  
    evas=[]                      # list containing scores from each episode
    # scores_window = deque(maxlen=100)  # last 100 scores
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
            # print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f}'.format(i_episode, frame, np.mean(scores_window)))
            # agent.qnetwork_local.save_weights('./QUANTUM/')
            if i_episode % 5 == 0:       
                eva=evaluate(eval_runs=3)
                evas.append(eva)
                print('evaluate=',eva)
                # agent.qnetwork_local.save_weights('./QUANTUM/qweight10')
            # if i_episode % 100 == 0:
            #     print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode,frame, np.mean(scores_window)))
            if i_episode % 100 == 0:    
                with open(f'./QUANTUM/qevas10_{i+20}','wb') as f:
                    pickle.dump(evas,f)
        
            i_episode +=1 
            state = env.reset()
            # state=state/np.array([1.0,1.0,8.0])
            step=0
            score = 0  
    with open(f'./QUANTUM/qevas10_{i+20}','wb') as f:
        pickle.dump(evas,f)
