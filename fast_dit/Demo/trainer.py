from fast_dit.Module.trainer import Trainer

def demo():
    model_file_path = './output/pretrain-v1/model_last.pt'

    trainer = Trainer(model_file_path)
    trainer.train()
    return True
