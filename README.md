# Quantum Reinforcement Learning with Cirq and TensorFlow Quantum (TFQ)

This repository contains a Quantum Reinforcement Learning (QRL) model developed using the quantum computing library Cirq and TensorFlow Quantum. The model is based on a variant of Deep Q-Networks (DQN), specifically the Normalized Advantage Function (NAF) algorithm, and is designed to interact and learn from its environment.

## Code Overview

The code is organized into several parts:

1. **Quantum Circuit Generation**: Functions `one_qubit_rotation`, `entangling_layer`, and `generate_circuit` for generating a quantum circuit.
2. **Quantum Circuit Execution**: Class `ReUploadingPQC` is a custom layer for TensorFlow that executes the quantum circuit in TensorFlow Quantum.
3. **Ornstein-Uhlenbeck Noise Process**: Class `OUNoise` defines a noise process, which adds exploration capability to the agent.
4. **Experience Replay Buffer**: Class `ReplayBuffer` manages the storage and sampling of past experiences of the agent.
5. **Normalized Advantage Function (NAF)**: Class `NAF` uses a Quantum Neural Network (QNN) to estimate the Q-value function and the policy.
6. **Deep Q-Network Agent**: Class `DQN_Agent` manages the action selection, learning, and updating processes of the Q-networks.

## Installation

The code requires the following packages:

- Cirq
- TensorFlow Quantum (TFQ)
- TensorFlow Probability
- Numpy

You can install these packages using pip:

```sh
pip install cirq tensorflow-quantum tensorflow-probability numpy
```

## Usage

Import the QRL model into your reinforcement learning framework and initialize an instance of the `DQN_Agent` class:

```python
from QRL import DQN_Agent

agent = DQN_Agent(observables, qubits, action_size, layer_size, BATCH_SIZE, BUFFER_SIZE, LR, LR2, TAU, GAMMA, UPDATE_EVERY, NUPDATES)
```

The agent can be used to interact with an environment (in the OpenAI Gym style) as follows:

```python
state = env.reset()
for t in range(max_t):
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    if done:
        break
```

Please refer to the comments in the code for more detailed explanations of the parameters and the workings of the algorithm.

## Contributing

Feel free to open an issue or submit a pull request if you have a suggestion or find a bug.

## License

This project is licensed under the terms of the MIT license.
