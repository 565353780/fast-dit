import sys
sys.path.append('../a-sdf/')
sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")

def demo():
    from fast_dit.Module.sampler import Sampler

    sampler = Sampler()
    sampler.sample()
    return True
