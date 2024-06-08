
import os
import pandas
csv_file = "submission_lyak.csv"
# csv_file = "submission_lyak_gcn.csv"
thresh = 0.25
ret = []
dic = {}

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
with open("submission_all.csv", "w", encoding="utf-8") as f:
    f.write("path1\tpath2\ttitle1\ttitle2\tscore\n")
    for line in df.values.tolist():
        score = line[-1]
        if float(score) > thresh:
            # print(line)
            id1, id2 = line[0], line[1]
            path1, t1 = dic[id1]
            path2, t2 = dic[id2]
            f.write(path1 + "\t" + path2 + "\t" + t1 + "\t" + t2 + "\t" + str(score) + "\n")
            # ret.append((id1, id2))
    f.close()




