get_ipython().system('pip install dostoevsky')
get_ipython().system('python -m dostoevsky download fasttext-social-network-model')
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer)
file = open("news.txt", "r")
lines = file.readlines()
results = model.predict(lines, k=2)
import pickle
f = open('data.pickle', 'wb')
pickle.dump(results, f)
f.close()
