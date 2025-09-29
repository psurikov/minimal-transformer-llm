from bpe_trainer import BpeTrainer

trainer = BpeTrainer()
vocat, merges = trainer.train(r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\tinystories_sample.txt""", 276, ["<|endoftext|>"])
