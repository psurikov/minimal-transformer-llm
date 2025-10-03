from bpe_trainer import BpeTrainer

trainer = BpeTrainer()
trainer.train(r"""./tests/fixtures/tinystories_sample.txt""", 276, ["<|endoftext|>"])
vocab = trainer.vocab
merges = trainer.merges
