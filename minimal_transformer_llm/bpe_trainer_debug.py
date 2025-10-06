from minimal_transformer_llm.bpe_trainer_sequential import BpeTrainer

trainer = BpeTrainer()
trainer.train(r"""./tests/fixtures/tinystories_sample.txt""", 276, ["<|endoftext|>"])
vocab = trainer.vocab
merges = trainer.merges
