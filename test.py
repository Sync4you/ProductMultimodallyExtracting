
import os
import pandas
csv_file = "submission_lyak.csv"
# csv_file = "submission_lyak_gcn.csv"
thresh = 0.25
ret = []
dic = {}

# import pickle
# f = open("submission_lyak.pkl", "rb")
# data = pickle.load(f)
# data.to_csv(csv_file)

with open("../train.csv", "r", encoding="utf-8") as f:
    d = f.readline()
    while d:
        line = d.replace("\n", "").split(',')
        id = line[0]
        path = line[1]
        title = line[2]
        dic[id] = (path, title)
        d = f.readline()
    f.close()


df = pandas.read_csv(csv_file, usecols=['posting_id','posting_id_target', 'pred'])
for line in df.values.tolist():
    score = line[-1]
    if float(score) > thresh:
        # print(line)
        id1, id2 = line[0], line[1]
        # path1, t1 = dic[id1]
        # path2, t2 = dic[id2]
        ret.append((id1, id2))

sets = []
for (id1, id2) in ret:
    f = False
    for k in sets:
        if id1 in k or id2 in k:
            k.add(id1)
            k.add(id2)
            f = True
    if f is False:
        a = set()
        a.add(id1)
        a.add(id2)
        sets.append(a)
# print(sets)
# cnt = 0
# for k in sets:
#     cnt += len(k)
# print(cnt)
with open("submission_cluster.csv", "w", encoding="utf-8") as f:
    for a in sets:
        for k in a:
            if dic.get(k) is not None:
                f.write(dic[k][1] + "\n")
            else:
                print(f"{k} is not existed!")
        f.write("\n\n")
    f.close()
# for (id1, id2) in ret:
#     f = False
#     for s in sets:
#         if id1 in s and id2 not in s:
#             s.append(id2)
#             f = True
#             break
#         elif id2 in s and id1 no  t in s:
#             s.append(id1)
#             f = True
#             break
#     if f is False:
#         sets.append([id1, id2])
#
# print(sets)
# cnt  = 0
# for s in sets:
#     cnt += len(s)
# print(cnt)
# with open("submission_cluster.csv", "w", encoding="utf-8") as f:
#     for lst in sets:
#         for id in lst:
#             f.write(dic[id][1] + "\n")
#         f.write("\n\n")
#     f.close()




