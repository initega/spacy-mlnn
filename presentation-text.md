# Problem
The goal of this project is to create a NER algorithm in Spacy able to recognize
a set of "Machine Learning" entities from the analysis of ML text, for example,
from the analysis of a ML paper.

## What are (Named) Entities?
An entity is a "real-world object" that is assigned a name. In other words,
an entity is basically a topic and there are **keywords** that are instances of
it.


### What are keywords?
Keywords could be any set of words that have relationship with a topic.

Keyword examples:
- Apple
- Knife
- Microwave

Named Entities:
- Fruit: **Apple**, Pineapple, Orange.
- Clutery: **Knife**, Spoon, Fork.
- Electrical appliance: **Microwave**, Freezer, Dishwasher.

#### Machine Learning related keywords

Keyword examples:

- Decision Tree
- Clustering
- Keras

Those three examples could be classified the next way:

- Supervised ML algorithm: **Decision Tree**, Random Forest, SVM
- Unsupervised ML algorithm: **Clustering**, KNN, Event detection
- ML Software: **Keras**, SkLearn, Tensorflow

## Name Entity Recognition Example
![cosas](/home/initega/Pictures/Screenshots/Screenshot-2019-12-11.png "hello d")

# Aproach

## Named Entities Used

Used Entities:
- SML: "Supervised Machine Learning Algorithms"
- USML: "Unsupervised Machine Learning Algorithms"
- MLS: "Machine Learning Software"
- NN: "Neural Networks"
- EVM: "Evatuation Methods" 
- OPM: "Optimization Methods" 
- MLP: "Machine Learning Preprocessing" 
- MLA: "Machine Learning Applications" 

    LABELS = ["EVM","MLA","MLP","MLS","NN","OPM","SML","USML"]

## Data example
In order to train the model, we had to create our own train data, taking pieces
of text and then explicitly showing where were the entities located and which
type it were.

    [
        (
            "We introduced a multilayer perceptron neural network (MLPNN) based
            classification model as a diagnostic decision support mechanism in the
            epilepsy treatment.",
            {"entities": [(27, 37, "NN")]}
        )
        ,
        ...
        ,
        (
            "The main issue in gradient descent is: how should we set the step
            size?",
            {"entities": [(18, 34, "OPM")]}
        )
    ]

As it is shown, each sample of data is formed by 3 main components:
- The text which is going to be used.
- The "entities" label, which indicates the entities inside the text.
- For each entity that is into the text, this will be represented giving its
  first character position, its last one and the entity related to that sequence
  of characters.

We used several shell scripts in order to get the locations for most of the
entries, but it isn't always possible.

That whole training set is an array of 245 of those kind of samples. That is
because we used 7 samples of text to train each entity keyword and there are 35
keywords defined.

## The algorithm
The first step is to define and set a pipeline for the model.
Once the pipe is set, the labels we have chosen are given to the model.

The following step is to start and give the training data to the model.

After being trained, the test data is given, and the predictions are made.

# The main function

    def ner(model=None, new_model_name="machine_learning", output_dir=None,
         n_iter=30, no_train=False, train=TRAIN_DATA, test=test_text,
         display=False, verbose=True):

# Evaluating accuracy
The last task we had to accomplish on this project was to design a validation
method to evaluate the accuracy of the method.

We did that designing a Cross Validation method for this specific problem.

What's more, we even intentionally tried overfit the model.

## Trying to overfit the model
### Testing the model with piece of text
Firstly, we tried a simple task, take a piece of text and give it to the model.
Check how good it is on different number of iterations and make conclusions.

With iterations: 30-150

#### Results

Accuracy is the ratio between the guesses made by the model
against the true entities which are in the model. Besides, false positives will
represent the times the model made a guess that is not between the real entities
of the model.  This count is not considered inside our definition of accuracy,
however, it is important to consider it while evaluating the general performance
of the model.

[image1]: "accuracy"
[image2]: "false-positives"
[image3]: "false-positives-2"

### Massive model learning loops
We tried to train a model using always the same input text.

After each training process, we would save the generated model and use that on
the next training process.

We did the same training process 80 times.

#### Results
The results were not what we expected, as that model had the same accuracy as
any once-trained model.

## Cross Validation
In order to to test the overall performance of our algorithm, we have used cross
validation.

As mentioned before, we trained the model with 245 samples, split in 35 groups
of 7 samples for a specific keyword. That means that cross validation cannot be
applied in any way, as the condition to find a keyword on a given text, is to
have learned about that keyword before.

So, our Cross Validation has 7 folds, each one of 35 samples with information
about different
keywords.

On Cross Validation, we also used several number of iterations in order to test
the performance with all them.

### Results
However, seems like on iterations below 30, because of not enough training, the
loss function is too poor, making the model too generalized, making it unable to
have a relevant accuracy.  As can be seen, the Cross Validation with 60
iterations is the best, but algorithms with surrounding iterations like 30 and
100 are also pretty good.  This behaviour’s cause is that, when the NER
algorithm reaches 30-60 iterations the loss function change from iteration to
iteration is not too relevant, which means that the loss function is starting to
converge.  It is also worth to mention that, in general, multi-word keywords are
less likely to be found that single-word keywords.  What’s more, this model is
able to find half of the keywords used, generally, pretty easily, but it has
problems with the other half.

# Conclusions
## Attempts to overfit the model
Directly adding more iterations to the NER parser trainer makes the parser think
that the huge amount of identical data it is receiving because of the high
number of iterations should be taken into account, eventually causing
overfitting.
However, when inputting the previously saved model into the next execution
constantly doesn’t generate overfitting, what it causes is what spaCy developers
call "catastrophic forgetting". In other words, when inputting a model, the
knowledge it had is overwritten in some way, so overfitting is not possible.
## Cross Validation
Regarding Cross Validation, it is true that on our case, the results were not
perfect, but that might be the result of having such small training data. 245
text samples seems good enough, but it must be taken into account that our task
was to recognize 35 keywords, which means that, on overall, only 7 samples of
text can be used for training each of them.
