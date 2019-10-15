import random
from collections import deque, namedtuple
from .per import PER

Transition = namedtuple(
    'Transition', ('state', 'action', 'done', 'next_state', 'reward'))
    #'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))



class ReplayMemory(object):
    def __init__(self, capacity, priority=False):
        self.capacity = capacity
        self.priority = priority
        if priority:
            self.memory = PER(capacity=capacity)
        else:
            self.memory = deque(maxlen=capacity)

    def push(self, *args, err=None):
        """Saves a transition."""
        if self.priority:
            assert err is not None, "Need to pass float error to add to priority memory"
            self.memory.add(err, Transition(*args))
        else:
            self.memory.append(Transition(*args))

    def sample(self, batch_size):
        if self.priority:
            batch, idx, is_weights = self.memory.sample(batch_size)
        else:
            batch = random.sample(self.memory, batch_size)
            idx = None
        batch = Transition(*zip(*batch))
        return batch, idx

    def update(self, idx, err):
        assert self.priority, "Cannot call this function if not priority memory"
        self.memory.update(idx, err)

    def batch_update(self, ids, errs):
        for idx, err in zip(ids, errs):
            self.update(idx, err)
        return

    def __len__(self):
        return len(self.memory)
