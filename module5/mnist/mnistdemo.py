import requests
import os

# ***** Student Demo Code ********
#  The student version.  This plus major_demo and LCM (below) should be given to the students ON DEMO DAY (not before).
# What the students must do to prepare for this: a) encapsulate their ANN in an object, b) write the method
# "blind_test" for their ANN class, and c) have all the MNIST files in the directory given by __mnist_path__

def student_tester(ann,cases,seed=1234,k=4):
    randomizer = LCG(seed)  # Valerij's randomizer (see below)
    images,_ = cases
    randomizer.shuffle(images)
    predictions = ann.blind_test(images)  # Students must write THIS method for their ANN
    return(score_classifications(seed,predictions,k=k))

def major_demo(ann,seed,dir):
    demo_cases = load_flat_text_cases('demo120_text.txt',dir)
    training_cases = load_flat_text_cases('all_flat_mnist_training_cases_text.txt',dir)
    test_cases = load_flat_text_cases('all_flat_mnist_testing_cases_text.txt',dir)
    print('TEST Results:')
    print('Training set: \n ',student_tester(ann,training_cases,seed,4))
    print('Testing set:\n ',student_tester(ann,test_cases,seed,4))
    print('Extra 120 set: \n ',student_tester(ann,demo_cases,seed,8))

def score_classifications(seed, classification,k=4):
    params = {"results": str(seed) + " " + str(classification), "raw": "1","k": k}
    resp = requests.post('http://folk.ntnu.no/valerijf/5/', data=params)
    return resp.text

# For reading flat cases from a TEXT file (as provided by Valerij). This is needed for the demo!  This also appears in
# mnist_basics, but I don't want to have to import that (student) file for the demo.

def load_flat_text_cases(filename,dir):
    f = open(dir + filename, "r")
    lines = [line.split(" ") for line in f.read().split("\n")]
    f.close()
    x_l = list(map(int, lines[0]))
    x_t = [list(map(int, line)) for line in lines[1:]]
    return x_t, x_l


# Linear Congruential Generator (LCG) - produces a list of random numbers and uses them to
# shuffle a list.  Written by Valerij, this should be shared between the student's code and
# Valerij's answer-checking system.  This insures that the same random seed produces the same
# shuffling order on the student's machine and on Valerij's.

class LCG:
    a = 1140671485
    c = 128201163
    m = 16777216

    def __init__(self, seed):
        self.reset(seed)

    def reset(self,seed): self.rand = seed

    def random(self):
        self.rand = (LCG.a*self.rand + LCG.c) % LCG.m
        return float(self.rand) / LCG.m

    def randint(self, limit):
        return int(limit * self.random())

    def shuffle(self, ary):
        for d in range(len(ary)-1, 0, -1):
          e = self.randint(d)
          if e == d:
                continue
          ary[d], ary[e] = ary[e], ary[d]
        return ary


# This is my test version of this protocol; the students don't need it.
def blind_tester(ann,cases,seed=1234,fid=None):
    randomizer = LCG(seed)  # Valerij's randomizer (see below)
    images,labels = cases
    randomizer.shuffle(images)
    predictions = ann.blind_test(images)  # Students must write THIS method for their ANN
    if fid:
        print(score_classifications(seed,predictions))
        # dump_predictions(fid,seed,predictions)
        # check_predictions(fid)
    randomizer.reset(seed) # insures same shuffling of images and labels
    randomizer.shuffle(labels)
    error = sum([(0 if p == l else 1) for p,l in zip(predictions,labels)])
    return (100 * error/len(predictions))


# Students get minor_demo (in mnist_basics.py) ahead of time to check that their system interfaces correctly with ours.


#  ********** OLD STUFF *****
# The file-based way to access Valerij's web-based evaluator

def dump_predictions(fid,seed,predictions):
    with open(fid,'w') as f:
        f.write(str(seed)+str(predictions))

# This sends the prediction file to Valerij's site for checking.
def check_predictions(fid):
    command = "curl -F \"data=@" + fid + "\" folk.ntnu.no/valerijf/5/"
    os.system(command)


