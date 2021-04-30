# 创建5gram的穷尽字典
dict_list = []
word_dict = {}
N = ["0","1","2","3","4"]
# 从3开始
index=3
for n in N:
    #print(n)
    A = N
    for a in A:
        #print(n+a)
        G = N
        for g in G:
            #print(n+a+g)
            C = N
            for c in C:
                #print(n+a+g+c)
                T =N
                for t in T:
                    #print(n+a+g+c+t)
                    #dict_list.append(n+a+g+c+t)
                    word_dict[n+a+g+c+t] = index
                    index += 1
#len(dict_list)
print("穷尽所有组合得到的字典长度：\n",len(word_dict))
# 把字典item转为list，再查看前5个
print("字典的前几个：\n", list(word_dict.items())[:5]) 


import pickle
def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
dict_path = "./data/word_dict_5gram.pkl"
save_obj(word_dict,dict_path)
print("保存字典：\n", dict_path)