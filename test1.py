import os

dic = {}
imgs = os.listdir('./images')
for csv_file in os.listdir('./csvs'):
    with open(os.path.join('./csvs', csv_file), "r", encoding="utf-8") as f:
        d = f.readline()
        while d:
            line = d.replace('\n', '').split('\t')
            img_id = line[0]
            if img_id + ".jpeg" not in imgs:
                print(img_id)
            else:
                tt = line[2]
                tt = tt.replace(",", " ")
                if dic.get(img_id, 0) == 0:
                    dic[img_id] = tt
                else:
                    if dic[img_id] != tt:
                        print("err:", dic[img_id], tt)
            d = f.readline()
#
print(len(dic.items()))
for i in imgs:
    ids = i.replace(".jpeg", "")
    if dic.get(ids, 0) == 0:
        print(ids)

with open("./train.csv", "w", encoding="utf-8") as f:
    f.write("posting_id,image,title\n")
    for k, v in dic.items():
        f.write(k + "," + k + ".jpeg," + v + "\n")
    f.close()
print("done")
