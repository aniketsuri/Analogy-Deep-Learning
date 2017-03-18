
# coding: utf-8

# Deep Learning Programming Assignment 2
# --------------------------------------
# Name: Aniket Suri
# Roll No.: 14CS10004
# 
# Submission Instructions:
# 1. Fill your name and roll no in the space provided above.
# 2. Name your folder in format <Roll No>_<First Name>.
#     For example 12CS10001_Rohan
# 3. Submit a zipped format of the file (.zip only).
# 4. Submit all your codes. But do not submit any of your datafiles
# 5. From output files submit only the following 3 files. simOutput.csv, simSummary.csv, analogySolution.csv
# 6. Place the three files in a folder "output", inside the zip.

# In[59]:

import gzip
import os
import tensorflow as tf
import numpy as np
import csv

from scipy import spatial

from math import*

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

n_nodes_hl1 = 10
n_nodes_hl2 = 50
n_nodes_hl3 = 50
n_input_nodes = 600
n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None, 600])
y = tf.placeholder('float')
x1 = tf.placeholder('float', [None, 600])

## paths to files. Do not change this
simInputFile = "Q1/word-similarity-dataset"
analogyInputFile = "Q1/word-analogy-dataset"
vectorgzipFile = "Q1/glove.6B.300d.txt.gz"
vectorTxtFile = "Q1/glove.6B.300d.txt"   # If you extract and use the gz file, use this.
analogyTrainPath = "Q1/wordRep/"
simOutputFile = "Q1/simOutput.csv"
simSummaryFile = "Q1/simSummary.csv"
anaSOln = "Q1/analogySolution.csv"
Q4List = "Q4/wordList.csv"




# In[ ]:

# Similarity Dataset
simDataset = [item.split(" | ") for item in open(simInputFile).read().splitlines()]
# Analogy dataset [['A','A','A','A','A']]
analogyDataset = [[stuff.strip() for stuff in item.strip('\n').split('\n')] for item in open(analogyInputFile).read().split('\n\n')]
	#[['A B','A B','A B','A B','A B','A B','a']]									#Probably a mistake here
print "analogyDataset",len(analogyDataset)
def vectorExtract(simD = simDataset, anaD = analogyDataset, vect = vectorgzipFile):
    simList = [stuff for item in simD for stuff in item] #All words of word-similarity-dataset
    analogyList = [thing for item in anaD for stuff in item[0:4] for thing in stuff.split()] #All(first 4 donno why) words of word-similarity-dataset
    print "analogyList",len(analogyList)
    simList.extend(analogyList)
    word_rep_words = []
    for subDirs in os.listdir(analogyTrainPath):
        for files in os.listdir(analogyTrainPath+subDirs+'/'):
            f = open(analogyTrainPath+subDirs+'/'+files).read().splitlines()
            for pair in f:
        	    word_rep_words.append(pair.split()[0])
        	    word_rep_words.append(pair.split()[1])
    simList.extend(word_rep_words)
    wordList = set(simList)
    #All the words in word-similarity-dataset + word-analogy-dataset
    print "wordList length : ",len(wordList)
    wordDict = dict()
    
    vectorFile = gzip.open(vect, 'r')
    for line in vectorFile:
        if line.split()[0].strip() in wordList:
            wordDict[line.split()[0].strip()] = line.split()[1:]
    
    
    vectorFile.close()
    print 'retrieved', len(wordDict.keys())
    return wordDict

# Extracting Vectors from Analogy and Similarity Dataset
validateVectors = vectorExtract()


# In[ ]:
filenames = []
# Dictionary of training pairs for the analogy task
trainDict = dict()
for subDirs in os.listdir(analogyTrainPath):
    for files in os.listdir(analogyTrainPath+subDirs+'/'):
        f = open(analogyTrainPath+subDirs+'/'+files).read().splitlines()
        trainDict[files] = f
        filenames.append(files)

print len(trainDict.keys())
print filenames
#print trainDict['08-DerivedFrom.txt']
#[[word\tword],[word word],...]

# In[58]:
def similarityTask(inputDS = simDataset, outputFile = simOutputFile, summaryFile=simSummaryFile, vectors=validateVectors):
    print 'hello world'

    """
    Output simSummary.csv in the following format
    Distance Metric, Number of questions which are correct, Total questions evalauted, MRR
    C, 37, 40, 0.61
    """

    """
    Output a CSV file titled "simOutput.csv" with the following columns

    file_line-number, query word, option word i, distance metric(C/E/M), similarity score 

    For the line "rusty | corroded | black | dirty | painted", the outptut will be

    1,rusty,corroded,C,0.7654
    1,rusty,dirty,C,0.8764
    1,rusty,black,C,0.6543


    The order in which rows are entered does not matter and so do Row header names. Please follow the order of columns though.
    """
    writer1 = csv.writer(open("simSummary.csv","w"))
    writer2 = csv.writer(open("simOutput.csv","w"))

    similarity = [0,0,0,0]
    count = 0
    total = 0
    mrr = 0.0
    #print inputDS[16][0]
    #print "Cosine:"
    for i in range(len(inputDS)):
    	if inputDS[i][0] not in vectors or inputDS[i][1] not in vectors or inputDS[i][2] not in vectors or inputDS[i][3] not in vectors or inputDS[i][4] not in vectors:
        	continue
        total +=1
        for j in range(1,len(inputDS[0])):
            if inputDS[i][0] in vectors and inputDS[i][j] in vectors:
                similarity[j-1] = 1 - spatial.distance.cosine(np.array(vectors[inputDS[i][0]],dtype=float), np.array(vectors[inputDS[i][j]],dtype=float))
                row = []
                row.append(i)
                row.append(inputDS[i][0])
                row.append(inputDS[i][j])
                row.append('C')
                row.append(similarity[j-1])
                #row = [i,inputDS[i][0],inputDS[i][i],'C',similarity[j-1]]
                writer2.writerow(row)
                #print i,",",inputDS[i][0],",",inputDS[i][j],",C,", similarity[j-1]   
            else:
                continue
        if similarity[0] == max(similarity):
            count = count + 1
        mrr += 1.0/(np.argmax(similarity)+1)
    mrr /= total       

    #print count," out of ",total," correct."  
    row = []
    row.append('C')
    row.append(count)
    row.append(total)
    row.append(mrr)
    writer1.writerow(row)
    
    #print "\n\nEuclidean Distance	\n"
    #dist = numpy.linalg.norm(a-b)
    Euclidean_distance = [0,0,0,0]
    count = 0
    total = 0
    mrr = 0.0
    #print inputDS[16][0]
    for i in range(len(inputDS)):
    	if inputDS[i][0] not in vectors or inputDS[i][1] not in vectors or inputDS[i][2] not in vectors or inputDS[i][3] not in vectors or inputDS[i][4] not in vectors:
        	continue
        total +=1
        for j in range(1,len(inputDS[0])):
            if inputDS[i][0] in vectors and inputDS[i][j] in vectors:
                Euclidean_distance[j-1] =np.linalg.norm(np.array(vectors[inputDS[i][0]],dtype=float) - np.array(vectors[inputDS[i][j]],dtype=float))
                row = []
                row.append(i)
                row.append(inputDS[i][0])
                row.append(inputDS[i][j])
                row.append('E')
                row.append(Euclidean_distance[j-1])
                #row = [i,inputDS[i][0],inputDS[i][i],'C',similarity[j-1]]
                writer2.writerow(row)

                #print i,",",inputDS[i][0],",",inputDS[i][j],",E,", Euclidean_distance[j-1]   
            else:
                continue
        if Euclidean_distance[0] == min(Euclidean_distance):
            count = count + 1
        mrr += 1.0/(np.argmin(Euclidean_distance)+1)
    mrr /= total       
    
    #print count," out of ",total," correct."  
    row = []
    row.append('E')
    row.append(count)
    row.append(total)
    row.append(mrr)
    writer1.writerow(row)
    
    #print "\n\nManhattan Distance	\n"
    #dist = numpy.linalg.norm(a-b)
    Manhattan_distance = [0,0,0,0]
    count = 0
    total = 0
    mrr = 0.0
    #print inputDS[16][0]
    for i in range(len(inputDS)):
    	if inputDS[i][0] not in vectors or inputDS[i][1] not in vectors or inputDS[i][2] not in vectors or inputDS[i][3] not in vectors or inputDS[i][4] not in vectors:
        	continue
        total +=1
        for j in range(1,len(inputDS[0])):
            if inputDS[i][0] in vectors and inputDS[i][j] in vectors:
                Manhattan_distance[j-1] = spatial.distance.cityblock(np.array(vectors[inputDS[i][0]],dtype=float) , np.array(vectors[inputDS[i][j]],dtype=float))
                row = []
                row.append(i)
                row.append(inputDS[i][0])
                row.append(inputDS[i][j])
                row.append('M')
                row.append(Manhattan_distance[j-1])
                #row = [i,inputDS[i][0],inputDS[i][i],'C',similarity[j-1]]
                writer2.writerow(row)
                #print i,",",inputDS[i][0],",",inputDS[i][j],",M,", Manhattan_distance[j-1]   
            else:
                continue
        if Manhattan_distance[0] == min(Manhattan_distance):
            count = count + 1
        mrr += 1.0/(np.argmin(Manhattan_distance)+1)
    mrr /= total       

    #print count," out of ",total," correct."  
    row = []
    row.append('M')
    row.append(count)
    row.append(total)
    row.append(mrr)
    writer1.writerow(row)

  
# In[ ]:
#validateVectors
#trainDict : traindict['filename'] = [[word\tword],[word word],[word word]...]
#filenames
#analogyDataset = [['A B','A B','A B','A B','A B','A B','a']]
def analogyTask(inputDS=analogyDataset,outputFile = anaSOln ): # add more arguments if required
    
    """
    Output a file, analogySolution.csv with the following entris
	    Query word pair, Correct option, predicted option    
    """
    
    pos = np.zeros((1,600))
    count = 0
    for i in range(len(filenames)):
        ##if count > 1000:
        #   break
    	file = trainDict[filenames[i]]
    	print "In file ",filenames[i]
    	l1 = 3
        for i1 in xrange(0,len(file)-l1,2):
            #if i1%500 == 0:
        	    #print i1,"pos : ",pos[:,0].shape," ",pos[0,:].shape
            #if count > 1000:
    	    #    break
    
            for i2 in xrange(i1+1,i1+l1+1):
            	if i2 >= len(file):
            		break
                if i1==i2:
                    continue
                    
                input_vector = []
                input_vector = np.array(input_vector,dtype=float)
                word1 = file[i1].split()[0]
                word2 = file[i1].split()[1]
                word3 = file[i2].split()[0]
                word4 = file[i2].split()[1]

                if word1 not in validateVectors or word2 not in validateVectors or word3 not in validateVectors or word4 not in validateVectors:
                    continue
                count += 1    
                input_vector = np.concatenate((input_vector,np.array(validateVectors[word2],dtype=float) - np.array(validateVectors[word1],dtype=float)))
                #print "input_vector :",np.array(validateVectors[word1],dtype=float).size
                #input_vector = np.concatenate((input_vector,np.array(validateVectors[word2],dtype=float)))
                input_vector = np.concatenate((input_vector,np.array(validateVectors[word4],dtype=float)-np.array(validateVectors[word3],dtype=float)))
                #input_vector = np.concatenate((input_vector,np.array(validateVectors[word4],dtype=float)))
                #print "input_vector :",input_vector.size
                if count%500 == 0:
                    print "Correct = ",cosine_similarity(input_vector[:300],input_vector[300:])# 1 - spatial.distance.cosine(input_vector[:300],input_vector[300:] )                
                pos = np.concatenate((pos,np.matrix(input_vector) ))
                
    pos = np.reshape(pos,(count+1,600))
    pos = pos[1:,:]
    print "pos : ",pos[:,0].shape," ",pos[0,:].shape
    
    #count = 50 # Remove this line soon
    posY = np.concatenate((np.ones((count,1)),np.zeros((count,1))),axis=1 )
    posY = np.matrix(posY)
    print "posY:",posY[:,0].shape," ",posY[0,:].shape

    
    neg = np.zeros((1,600))
    all_words = []
    count = 0
    for file_name,file_content in trainDict.iteritems():
        for pair in file_content:
            all_words.append([file_name,pair])
    status = 1
    for k in xrange(0,len(all_words)):
    	count2 = 0
        for j in xrange(k+1,len(all_words)):
            if(all_words[j][0] == all_words[k][0]):
                continue
            m = all_words[j][1].split()
            n = all_words[k][1].split()
            if m[0] not in validateVectors or m[1] not in validateVectors or n[0] not in validateVectors or n[1] not in validateVectors:
                break

            input_vector = []
            input_vector = np.array(input_vector,dtype=float)
            input_vector = np.concatenate((input_vector,np.array(validateVectors[m[1]],dtype=float)-np.array(validateVectors[m[0]],dtype=float)))
            #input_vector = np.concatenate((input_vector,np.array(validateVectors[m[1]],dtype=float)))
            input_vector = np.concatenate((input_vector,np.array(validateVectors[n[1]],dtype=float)-np.array(validateVectors[n[0]],dtype=float)))
            #input_vector = np.concatenate((input_vector,np.array(validateVectors[n[1]],dtype=float)))
            neg = np.concatenate((neg,np.matrix(input_vector) ))
            if count%500 == 0:
                print count,"neg : ",neg[:,0].shape," ",neg[0,:].shape
            count = count + 1
            count2 += 1
            if(count2 == 2):
            	break
            if(count > 30000):
                status = 0
                break
        if(status == 0):
            break
    neg = np.reshape(neg,(count+1,600))
    neg = neg[1:,:]
    print "neg : ",neg[:,0].shape," ",neg[0,:].shape
    
    #count = 50 # Remove this line soon
    negY = np.concatenate((np.zeros((count,1)),np.ones((count,1))),axis=1 )
    negY = np.matrix(negY)
    print "negY:",negY[:,0].shape," ",negY[0,:].shape
    #print negY[0]
    
    trainY = np.concatenate((posY,negY))
    trainX = np.concatenate((pos,neg))

    perm = np.random.permutation(trainX.shape[0])
    trainX = trainX[perm]
    trainY = trainY[perm]

    print "trainX:",trainX[:,0].shape," ",trainX[0,:].shape     
    print "trainY:",trainY[:,0].shape," ",trainY[0,:].shape
    #print trainY[0]
    #print trainY[51]
    
    testX = np.zeros((1,600))
    testY = [[0,0]]
    #analogyDataset = [['A B','A B','A B','A B','A B','A B','a']]
    count = 0
    count_correct=0
    count_uncorrect = 0
    for item in analogyDataset:
    	if item[0].split()[0] not in validateVectors or item[0].split()[1] not in validateVectors \
    	or item[1].split()[0] not in validateVectors or item[1].split()[1] not in validateVectors \
    	or item[2].split()[0] not in validateVectors or item[2].split()[1] not in validateVectors \
    	or item[3].split()[0] not in validateVectors or item[3].split()[1] not in validateVectors \
    	or item[4].split()[0] not in validateVectors or item[4].split()[1] not in validateVectors \
    	or item[5].split()[0] not in validateVectors or item[5].split()[1] not in validateVectors:
            continue
        for i,ch in [(1,'a'),(2,'b'),(3,'c'),(4,'d'),(5,'e')]:
            word1 = item[0].split()[0]
            word2 = item[0].split()[1]
            word3 = item[i].split()[0]
            word4 = item[i].split()[1]        	
            input_vector = []
            input_vector = np.array(input_vector,dtype=float)
            input_vector = np.concatenate((input_vector,np.array(validateVectors[word2],dtype=float)-np.array(validateVectors[word1],dtype=float)))
            #input_vector = np.concatenate((input_vector,np.array(validateVectors[word2],dtype=float)))
            input_vector = np.concatenate((input_vector,np.array(validateVectors[word4],dtype=float)-np.array(validateVectors[word3],dtype=float)))
            #input_vector = np.concatenate((input_vector,np.array(validateVectors[word4],dtype=float)))
            count += 1
            testX = np.concatenate((testX,np.matrix(input_vector) ))
            if item[6] == ch:
                testY.append([1,0])
                count_correct += 1
                print "Correct = ",cosine_similarity(input_vector[:300],input_vector[300:])# 1 - spatial.distance.cosine(input_vector[:300],input_vector[300:] )
            else:
                testY.append([0,1])
                count_uncorrect += 1
                #print "Incorrect = ",cosine_similarity(input_vector[:300],input_vector[300:])# 1 - spatial.distance.cosine(input_vector[:300],input_vector[300:] )

    testX = np.reshape(testX,(count+1,600))
    testX = testX[1:,:]
    testY = np.matrix(testY)
    testY = np.reshape(testY,(count+1,2))    
    testY = testY[1:,:]

    print count_correct," out of ",count_uncorrect+count_correct
    print "testX :",testX[:,0].size," ",testX[0,:].size
    print "testY :",testY[:,0].size," ",testY[0,:].size
    '''
    np.savez('weights.npz', a=trainX, b=trainY, c=testX, d=testY)
    '''
    data = np.load('weights.npz')
    trainX = data['a']
    trainY = data['b']
    testX = data['c']
    testY = data['d']
    print "testX : ",testX[:,0].size," ",testX[0,:].size
    print "testY : ",testY[:,0].size," ",testY[0,:].size

    accuracy = train_neural_network(x,trainX,trainY,testX,testY)


    #accuracy = 0.0
    return accuracy #return the accuracy of your model after 5 fold cross validation



# In[60]:

def derivedWOrdTask(inputFile = Q4List):
    print 'hello world'
    
    """
    Output vectors of 3 files:
    1)AnsFastText.txt - fastText vectors of derived words in wordList.csv
    2)AnsLzaridou.txt - Lazaridou vectors of the derived words in wordList.csv
    3)AnsModel.txt - Vectors for derived words as provided by the model
    
    For all the three files, each line should contain a derived word and its vector, exactly like 
    the format followed in "glove.6B.300d.txt"
    
    word<space>dim1<space>dim2........<space>dimN
    charitably 256.238 0.875 ...... 1.234
    
    """
    
    """
    The function should return 2 values
    1) Averaged cosine similarity between the corresponding words from output files 1 and 3, as well as 2 and 3.
    
        - if there are 3 derived words in wordList.csv, say word1, word2, word3
        then find the cosine similiryt between word1 in AnsFastText.txt and word1 in AnsModel.txt.
        - Repeat the same for word2 and word3.
        - Average the 3 cosine similarity values
        - DO the same for word1 to word3 between the files AnsLzaridou.txt and AnsModel.txt 
        and average the cosine simialities for valuse so obtained
        
    """
    cosVal1,cosVal2 = 0,0
    return cosVal1,cosVal2



def load_mnist():
    data_dir = '../data'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX, trY, teX, teY




def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_input_nodes, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    #hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    #                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    #hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    #                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    #l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    #l2 = tf.nn.relu(l2)

    #l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    #l3 = tf.nn.relu(l3)

    output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x,trainX_,trainY_,testX_,testY_):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        trainX, trainY, testX, testY = trainX_,trainY_,testX_,testY_#load_mnist()
        trainX = np.matrix(trainX)

        testX = np.matrix(testX)

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for k in xrange(0,trainX[:,0].size,batch_size):
            	epoch_x,epoch_y = trainX[k:k+batch_size,:], trainY[k:k+batch_size,:]
            	_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            	epoch_loss += c	

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_test = accuracy.eval({x:testX, y:testY})
        accuracy_train = accuracy.eval({x:trainX, y:trainY})
        print('Accuracy_test:',accuracy_test)
        print('Accuracy_train:',accuracy_train)
        return accuracy_test

        

# In[ ]:

def main():
    #similarityTask()
    anaSim = analogyTask()
    derCos1,derCos2 = derivedWOrdTask()
    #train_neural_network(x) #Call this function when training data is set
    
if __name__ == '__main__':
    main()
