import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dqn_agent import DDQNAgent

def train():
    env = gym.make('ALE/Breakout-v5', frameskip=4, repeat_action_probability=0, render_mode='rgb_array')
    agent = DDQNAgent(env.action_space.n, batch_size=32)
    agent.load('./DDQNAgent-final-v1.chkpt')

    writer = SummaryWriter('runs/ddqn-nature2')

    return_arr = []
    # num_timesteps = 1_000_000
    num_timesteps = 1_000
    epochs = 10
    steps = 0
    episode = 1

    for epoch in range(epochs):
        s, info = env.reset()
        old_lives = info['lives']
        # shoot ball so that the ball becomes visible
        s = env.step(1)
        s = s[0]
        s = agent.preprocess(s)
        s = torch.stack([s, s, s, s])
        print('===================================')
        print(f'Epoch: {epoch}')
        ep_steps = 0
        Return = 0
        for step in range(num_timesteps):
            a = agent.act(s)
            (s_, r, d, t, info) = env.step(a)
            s_ = agent.preprocess(s_)
            s_ = torch.stack([*s[1:], s_])

            agent.cache((s, a, r, s_, d or t))

            s = s_

            Return += r
            ep_steps += 1
            agent.learn()

            steps += 1

            # shoot ball
            if old_lives > info['lives']:
                s = env.step(1)
                s = s[0]
                s = agent.preprocess(s)
                s = torch.stack([s, s, s, s])
                old_lives = info['lives']

            if d or t:
                print('-----------------------')
                print(f'Episode: {episode}')
                print(f'Epsilon: {agent.e}')
                print(f'Step: {steps}, Return: {Return}\n')
                writer.add_scalar('Return', Return, steps)
                writer.add_scalar('Steps per episode', ep_steps, steps)
                writer.add_scalar('Epsilon', agent.e, steps)
                writer.add_scalar('Total episodes', episode, steps)
                s, info = env.reset()
                # shoot ball so that the ball becomes visible
                s = env.step(1)
                s = s[0]
                s = agent.preprocess(s)
                s = torch.stack([s, s, s, s])
                return_arr.append(Return)
                Return = 0
                ep_steps = 0
                episode += 1

            # sync main and target network after agent.r steps
            if steps % agent.r == 0:
                agent.update_target_network()
                print(f'Step: {steps}, Return: {Return}')

        # agent.save(f'final-{steps}')

    torch.save(
        dict(model=agent.Q.state_dict()),
        f'DDQNAgent-final-v2.chkpt',
    )

    return {"Model": "DDQN", "Latest average reward": np.mean(return_arr[-num_timesteps:]), "Steps": steps,
            "Episodes": episode}

    pass

if __name__ == "__main__":
    train()
