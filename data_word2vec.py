import torch
import torch.nn as nn
import torch.nn.functional as F
from criterion_torch import NLLLoss
from optimizer_torch import SGD

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
pairs_input_lang = ''
trigrams = [([pairs_input_lang[i], pairs_input_lang[i + 1]], pairs_input_lang[i + 2])
            for i in range(len(pairs_input_lang) - 2)]

vocab = set(pairs_input_lang)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        # 첫번째. 모델에 넣어줄 입력값을 준비합니다. (i.e, 단어를 정수 인덱스로
        # 바꾸고 파이토치 텐서로 감싸줍시다.)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # 두번째. 토치는 기울기가 *누적* 됩니다. 새 인스턴스를 넣어주기 전에
        # 기울기를 초기화합니다.
        model.zero_grad()

        # 세번째. 순전파를 통해 다음에 올 단어에 대한 로그 확률을 구합니다.
        log_probs = model(context_idxs)

        # 네번째. 손실함수를 계산합니다. (파이토치에서는 목표 단어를 텐서로 감싸줘야 합니다.)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # 다섯번째. 역전파를 통해 기울기를 업데이트 해줍니다.
        loss.backward()
        optimizer.step()

        # tensor.item()을 호출하여 단일원소 텐서에서 숫자를 반환받습니다.
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # 반복할 떄마다 손실이 줄어드는 것을 봅시다!

# "beauty"와 같이 특정 단어에 대한 임베딩을 확인하려면,
print(model.embeddings.weight[word_to_ix["beauty"]])