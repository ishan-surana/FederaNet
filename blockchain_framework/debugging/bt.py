import blockchain
import pickle

blockchain = pickle.load(open("blockchain.pkl", "rb"))
print(blockchain.get_chain())