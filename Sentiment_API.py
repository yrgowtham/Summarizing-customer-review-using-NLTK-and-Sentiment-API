import requests
url = "http://text-processing.com/api/sentiment/"
def app(sentenced,i):
    pos=0
    neg=0
    positive=[]
    negative=[]
    for each in sentenced:
        print each
        respose = requests.post(
        url,
        data = {
        'text':each,
        }
        )
        data = respose.json()
        if data["probability"]["pos"]>0.5:
            pos=pos+1
            positive.append(each)
        elif data["probability"]["neg"]>0.5:
            neg=neg+1
            negative.append(each)

    print(positive)
    print("\n count of positive")
    print(pos)
    print("\n actual positive review")
    for act_pos_sent in positive:
        print(act_pos_sent)
    print("\ncount of neg")
    print(neg)
    print("\n negative review")
    for act_neg_sent in negative:
        print(act_neg_sent)
# data = json.load(test.text)
# print(data)
