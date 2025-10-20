import torch
from minimal_transformer_llm.softmax import softmax

def cross_entropy(o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # o: logits tensor — each vector along the last dimension represents
    #    the unnormalized scores (logits) over the entire vocabulary
    #    for predicting the next token at that position.
    #
    # x: target tensor — integer indices of the correct next tokens.
    #    It has the same leading dimensions as o (without the last vocab dimension).
    #
    # r: loss tensor — per-position (per-token) scalar loss values.

    # dims:
    # o (logits) : " ... sequence_length vocab_size"
    # x (targets): " ... sequence_length"
    # r (output) : " ... "

    # assume that pi is the probability distribution over vocab, telling how likely each token is to be after the current position i
    # if x is an index of the correct token that should be after, then pi[xi] is the actual probability we got (it could be low or high, this is what we want to improve)
    # the formula for the probability is what we calculate with softmax:
    # pi = (exp(oi)/sum_exp(oj)) where j is every element
    # the cross entropy is calculated as a log for the probability of a particular item - xi
    # cross_entropy i = -log (pi)
    # putting this all together, we get:
    # cross_entropy i = -log (exp(oi)/sum_exp(oj))
    # for numerical stability, it makes sense to subtract the largest element from all elements in logits, assume it is 'm':
    # cross_entropy i = -log (exp(oi - m)/sum_exp(oj - m))
    # log and exp can cancel each other out, so we get this:
    # cross_entropy i = log (sum_exp(oj - m)) - oi + m

    largest = torch.max(o, dim=-1, keepdim=True).values
    shifted = o - largest
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    expected_logit = o.gather(-1, x.unsqueeze(-1)).squeeze(-1)
    loss = logsumexp - expected_logit + largest.squeeze(-1)
    ave = torch.mean(loss, dim = -1)
    return ave