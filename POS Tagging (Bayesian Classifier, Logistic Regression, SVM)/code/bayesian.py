import nltk

## Load data
file_train = open('/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/midterm project/train.txt')
train = file_train.read()
file_test = open('/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/midterm project/unlabeled_test_test.txt')
test = file_test.read()

## Data preprocessing
data_list = train.split("\n")
data_list2 = test.split("\n")

train = []
for line in data_list:
    line_list = line.split(" ")
    if len(line_list) > 1:
        train.append(tuple(line_list[:2]))

test = []
for line in data_list2:
    line_list = line.split(" ")
    test += line_list
#print(data2)

## Feature extraction
def featureExtraction(n):
    return {'word': n}

featured_data = [(featureExtraction(word), tag) for word, tag in train]

## Train Bayesian Classifier
model = nltk.NaiveBayesClassifier.train(featured_data)

## Predict
predict = [model.classify(featureExtraction(word)) for word in test]

for word, tag in zip(test, predict):
    if word == "":
        print(word)
    else:
        print(f"{word}, {tag}")

## Accuracy
test_accuracy = []
for word, tag in zip(test, predict):
    test_accuracy.append((featureExtraction(word), tag))
        
accuracy = nltk.classify.accuracy(model, test_accuracy)
print(accuracy)

'''
len = 0
test_file = open('/Users/siheonjung/Desktop/psu/fall 2023/cmpsc448/midterm project/submit/Bayesian Classifier/SiheonJung.test.txt', 'w')
for word, tag in zip(test, predict):
    if word == "":
        if len < 15972:
            test_file.write("\n")
            len += 1
    
    else:
        test_file.write(word + ' ' + tag + '\n')
        len += 1
        
test_file.close()
'''