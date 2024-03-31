import numpy as np
import torch
from torch import nn
from collections import deque
from copy import deepcopy
import cv2
import random


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class DQN(nn.Module):
    def __init__(self, num_action_space):
        super().__init__()
        self.m = num_action_space

        # CNN to read and extract the important features from the input images of breakout

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        ##
        self.fcnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.m),
        )

        self.loss_fn = nn.HuberLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, amsgrad=True)

    def forward(self, X):
        return self.fcnn(self.features(X))

    pass


class DQNAgent(nn.Module):
    def __init__(
            self,
            num_action_space,
            batch_size,

            memory_size=80_000,
            gamma=0.99,
            update_rate=10_000,
            epsilon=1,
            min_epsilon=0.1,
            max_epsilon=1,
            epsilon_greedy_rate=1_000_000

    ):
        super().__init__()
        self.m = num_action_space
        self.e = epsilon
        self.e_min = min_epsilon
        self.e_max = max_epsilon
        self.e_rate = (self.e_max - self.e_min) / epsilon_greedy_rate
        self.b = batch_size
        self.Q = DQN(self.m).to(device)
        self.T = deepcopy(self.Q)
        self.g = gamma
        self.r = update_rate

        # no need to track target network's gradients
        for p in self.T.parameters():
            p.requires_grad = False

        self.update_target_network()
        self.mc = memory_size
        self.memory = deque(maxlen=self.mc)

    def preprocess(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)[32:, 8:152]
        state = cv2.resize(src=state, dsize=(84, 84)) / 255.
        return torch.from_numpy(state.astype(np.float32)).to(device)

    def update_target_network(self):
        # sync main and target network
        self.T.load_state_dict(self.Q.state_dict())

    def act(self, state, ep=None):
        # epsilon greedy strategy:
        # we select random action with epsilon prob
        # and follow policy otherwise
        if np.random.uniform(0, 1) < (ep or self.e):
            action = np.random.randint(self.m)
        else:
            action = self.Q(state.unsqueeze(0))[0].argmax().item()

        # here we decrease epsilon value after every step
        # by a constant factor of e_rate which is (e_max - e_min) / epsilon_greedy_frame
        if not ep:
            self.e -= self.e_rate
            self.e = max(self.e_min, self.e)
        return action

    def cache(self, exp):
        # store data in replay buffer
        s, a, r, s_, d = exp
        a = torch.tensor(a).to(device)
        r = torch.sign(torch.tensor(r)).to(device)
        d = torch.tensor(d).to(device)
        self.memory.append((s, a, r, s_, d))

    def memory_size(self):
        return len(self.memory)

    def sample_memory(self):
        return random.sample(self.memory, self.b)

    def update_epsilon(self, e, e_min, e_max, epsilon_greedy_rate):
        self.e, self.e_min, self.e_max = e, e_min, e_max
        self.e_rate = (self.e_max - self.e_min) / epsilon_greedy_rate

    def learn(self):
        if self.memory_size() < self.b: return

        exps = self.sample_memory()
        s, a, r, s_, d = map(torch.stack, zip(*exps))

        pred_q = self.Q(s)[np.arange(self.b), a]

        # bellman backup for DQN algorithm, here action selection and q value
        # computation is both done using target network
        target_q = r + (1 - d.float()) * self.g * self.T(s_).max(axis=1).values

        # backprop
        self.Q.optimizer.zero_grad()
        loss = self.Q.loss_fn(pred_q, target_q)
        loss.backward()
        self.Q.optimizer.step()

    def save(self, steps):
        torch.save(
            dict(model=self.Q.state_dict(), exploration_rate=self.e),
            f'DQNAgent-{steps}.chkpt',
        )

    def load(self, path):
        model_state_dict = torch.load(path)['model']
        self.Q.load_state_dict(model_state_dict)
        self.T.load_state_dict(model_state_dict)
        self.e = 0.3  # 或者你可以手动设置一个默认值

    pass


class DDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn(self):
        if self.memory_size() < self.b: return

        exps = self.sample_memory()
        s, a, r, s_, d = map(torch.stack, zip(*exps))

        # bellman backup for DDQN algorithm, here action selection is done
        # using main network and q value computation is done using target network
        pred_q = self.Q(s)[np.arange(self.b), a]
        target_q = r + (1 - d.float()) * self.g * self.T(s_)[
            np.arange(self.b),
            self.Q(s_).argmax(axis=1)
        ]

        # backprop
        self.Q.optimizer.zero_grad()
        loss = self.Q.loss_fn(pred_q, target_q)
        loss.backward()
        self.Q.optimizer.step()

    def save(self, steps):
        torch.save(
            dict(model=self.Q.state_dict(), exploration_rate=self.e),
            f'DDQNAgent-{steps}.chkpt',
        )
    pass
