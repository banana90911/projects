from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

## Load data
file_train = open('/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/midterm project/train.txt')
train = file_train.read()
file_test = open('/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/midterm project/unlabeled_test_test.txt')
test = file_test.read()

## Data preprocessing
data_list = train.split("\n")
data_list2 = test.split("\n")

train_x = []
train_y = []
word_index = {}
i = 0
for line in data_list:
    line_list = line.split(" ")
    if len(line_list) > 1:
        train_x.append(line_list[0])
        train_y.append(line_list[1])
    for word in line_list:
        if word not in word_index:
            word_index[word] = i
            i += 1

test = []
for line in data_list2:
    line_list = line.split(" ")
    test += line_list

vectorizer = CountVectorizer()
featured_word = vectorizer.fit_transform(train_x)

## Train Logistic Regression model
model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
model.fit(featured_word, train_y)

## Predict
test_vector = vectorizer.transform(test)
predict = model.predict(test_vector)

for word, tag in zip(test, predict):
    if word == "":
        print(word)
    else:
        print(f"{word}, {tag}")

len = 0
test_file = open('/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/midterm project/submit/Logistic Regression/SiheonJung.test.txt', 'w')
for word, tag in zip(test, predict):
    if word == "":
        if len < 15972:
            test_file.write("\n")
            len += 1
    else:
        test_file.write(word + ' ' + tag + '\n')
        len += 1
test_file.close()