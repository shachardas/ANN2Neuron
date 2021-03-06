# based on https://github.com/cuhkrlcourse/RLexample/blob/master/policygradient/reinforce.py

import argparse
import imp
from platform import architecture
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import config


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--do-learn', action='store_true', default=False,
                    help='use a leaning model or a predefined configuration (default: False - predefined)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)


def Binarize(tensor, include_zero = False):
    if include_zero:
        return ((tensor+0.5).sign()+(tensor-0.5).sign())/2
    else:
        return tensor.sign()

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class PositiveBinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(PositiveBinarizeLinear, self).__init__(*kargs, **kwargs)
 
    def forward(self, input):
        zero = torch.zeros_like(input.data)
        input.data = torch.where(input.data > 0, input.data, zero)
        input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = BinarizeLinear(4, 10, bias = False)
        self.affine2 = BinarizeLinear(10, 100, bias = False)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        action_scores = self.affine2(x)
        return action_scores


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def sumBits(b, bits):
    #mask = 2**torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    mask = torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

# convert the results given by the model to two probability distributions
def TranslateResults(modelRes):
    modelRes = (Binarize(modelRes[0])+1)/2
    l = int(len(modelRes)/2)
    actions = torch.ones(1,2)
    actions[0][0] = sumBits(modelRes[:l], l)
    actions[0][1] = sumBits(modelRes[l:], l)
    return F.softmax(actions)

def direct_policy(obs):
    theta, w = obs[2:4] 
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1


"""
predefined model:
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1
"""
class Predefined_policy(nn.Module):
    def __init__(self):
        super(Predefined_policy, self).__init__()
        self.fc = BinarizeLinear(4, 2, bias = False)
        self.fcp = PositiveBinarizeLinear(2, 1, bias = False)
        self.fc.weight = nn.Parameter(torch.tensor([[1.0,0.0,-1.0,0],[0.0,1.0,0,-1.0]]))
        self.fcp.weight = nn.Parameter(torch.tensor([[1.0,1.0]]))
        

        self.saved_log_probs = []
        self.rewards = []

    def parseInput(self, x):
        theta, w = x[0][2:4]
        res = torch.zeros([1,4])
        res[0][0] = float(theta > 0)
        res[0][1] = float(w > 0)
        res[0][2] = float(abs(theta) < 0.03)
        res[0][3] = float(abs(theta) >= 0.03)
        return res

    def forward(self, x):
        x = self.parseInput(x)
        x = self.fc(x)
        action_scores = self.fcp(x)
        return action_scores

predefined_policy = Predefined_policy()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    if args.do_learn:
        res = policy(state)
        probs = TranslateResults(res)
        m = Categorical(probs)
        sample = m.sample()
        action = m.log_prob(sample)
    else:
        action = predefined_policy(state)
        test = direct_policy(state[0])
        if action != test:
            print(action, test)
            print(state)
            exit()
        sample = torch.tensor([int(action)])
    policy.saved_log_probs.append(action)
    return sample.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    if args.do_learn:
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    print("started")
    running_reward = 10
    if not args.do_learn:
        torch.save(predefined_policy.state_dict(), config.TRAINED_MODELS_DIR + "predefined-CartPole" + ".pt")
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            #action = theta_omega_policy(state)#select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()