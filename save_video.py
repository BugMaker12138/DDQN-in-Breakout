import gym
from dqn_agent import DDQNAgent
import torch
def save_video():
    env = gym.make('ALE/Breakout-v5', frameskip=4, repeat_action_probability=0, render_mode='rgb_array')
    agent = DDQNAgent(env.action_space.n, batch_size=32)
    agent.load('./DDQNAgent-final-v1.chkpt')

    rec_env = gym.wrappers.RecordVideo(env, './ddqn')
    rec_env.metadata['render_fps'] = 24

    episodes = 1
    for e in range(episodes):
        Return = 0
        s, _ = rec_env.reset()
        s = rec_env.step(1)
        s = s[0]
        s = agent.preprocess(s)
        s = torch.stack([s, s, s, s])
        steps = 10_000
        for i in range(steps):
            a = agent.act(s)
            (s_, r, d, t, _) = rec_env.step(a)
            s_ = agent.preprocess(s_)
            s_ = torch.stack([*s[1:], s_])
            s = s_
            Return += r

            if d or t:
                print('done d')
                break

        if (e + 1) % 10 == 0:
            print(f"Episode: {e + 1}, Return: {Return}")

    pass

if __name__ == "__main__":
    save_video()
