from torchtext.data.metrics import bleu_score

def Bleu_socre(candidate_corpus, referencese_corpus):
    return bleu_score(candidate_corpus, referencese_corpus)