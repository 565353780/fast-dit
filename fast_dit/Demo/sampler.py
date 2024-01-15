import sys
sys.path.append('../a-sdf/')

def demo():
    from fast_dit.Module.sampler import Sampler

    sampler = Sampler()
    sampler.sample()
    return True
