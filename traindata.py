TRAIN_DATA = [
# accuracy
    (
        "If we have a separate test set we can evaluate performance on this in order to estimate the accuracy of our method.",
        {"entities": [(92, 100, "EVM")]}
    ),
    (
        "The accuracy of an MC approximation increases with sample size.",
        {"entities": [(4, 12, "EVM")]}
    ),
    (
        "Accuracy of Monte Carlo approximation",
        {"entities": [(0,8,"EVM")]}
    ),
    (
        "where K is chosen based on some tradeoff between accuracy and complexity.",
        {"entities": [(49,57,"EVM")]}
    ),
    (
        "In high dimensional problems we might prefer a method that only depends on a subset of the features for reasons of accuracy and interpretability.",
        {"entities": [(115, 123, "EVM")]}
    ),
    (
        "In machine learning we often care more about predictive accuracy than in interpreting the parameters of our models.",
        {"entities": [(56, 64, "EVM")]}
    ),
    (
        "However accuracy is not the only important factor when choosing a method.",
        {"entities": [(8, 16, "EVM")]}
    ),
# cross validation
    (
        "A simple but popular solution to this is to use cross validation (CV). The idea is simple: we split the training data into K folds; then for each fold k ∈ {1 . . .  K} we train on all the folds but the k’th and test on the k’th in a round-robin fashion",
        {"entities": [(48, 64, "EVM")]}
    ),
    (
        "It is common to use K = 5; this is called 5-fold CV. If we set K = N  then we get a method called leave-one out cross validation or LOOCV.",
        {"entities": [(112, 128, "EVM")]}
    ),
    (
        "We can use methods such as cross validation to empirically choose the best method for our particular problem.",
        {"entities": [(27, 43, "EVM")]}
    ),
    (
        "The principle problem with cross validation is that it is slow since we have to fit the model multiple times.",
        {"entities": [(27, 43, "EVM")]}
    ),
    (
        "Use cross validation to choose the strength of the 2 regularizer.",
        {"entities": [(4, 20, "EVM")]}
    ),
    (
        "In supervised learning we can always use cross validation to select between non-probabilistic models of different complexity but this is not the case with unsupervised learning.",
        {"entities": [(41, 57, "EVM")]}
    ),
    (
        "This is likely to be much faster than cross validation especially if we have many hyper-parameters (e.g. as in ARD).",
        {"entities": [(38, 54, "EVM")]}
    ),
# likelihood
    (
        "Use grid-search over a range of K’s using as an objective function cross-validated likelihood.",
        {"entities": [(83, 93, "EVM")]}
    ),
    (
        "That is they can use likelihood models of the form p(x t:t+l |z t = k d t = l) which generate l correlated observations if the duration in state k is for l time steps.",
        {"entities": [(21, 31, "EVM")]}
    ),
#    (
#        "the likelihood for the binomial sampling model is the same as the likelihood for the Bernoulli model.",
#        {"entities": [(4,14,"EVM"),(66,76,"EVM")]}
#    ),
    (
        "posterior is a combination of prior and likelihood.",
        {"entities": [(40,50,"EVM")]}
    ),
    (
        "It makes more sense to try to approximate the smoothed distribution rather than the backwards likelihood term.",
        {"entities": [(94, 104, "EVM")]}
    ),
    (
        "The gradient of the log likelihood can be rewritten as the expected feature vector according to the empirical distribution minus the model’s expectation of the feature vector.",
        {"entities": [(24, 34, "EVM")]}
    ),
    (
        "We call this algorithm stochastic maximum likelihood or SML.",
        {"entities": [(42, 52, "EVM")]}
    ),
    (
        "Nevertheless coordinate descent can be slow. An alternative method is to update all the parameters at once by simply following the gradient of the likelihood.",
        {"entities": [(147, 157, "EVM")]}
    ),
# recall
    (
        "From this table we can compute the true positive rate (TPR) also known as the sensitivity, recall or hit rate.",
        {"entities": [(90, 96, "EVM")]}
    ),
    (
        "For a ﬁxed threshold, one can compute a single precision and recall value.",
        {"entities": [(61,67,"EVM")]}
    ),
    (
        "Precision measures what fraction of our detections are actually positive and recall measures what fraction of the positives we actually detected.",
        {"entities": [(77, 83, "EVM")]}
    ),
    (
        "A precision recall curve is a plot of precision vs recall as we vary the threshold",
        {"entities": [(12,18,"EVM"),(51,57,"EVM")]}
    ),
    (
        "recall measures what fraction of the positives we actually detected.",
        {"entities": [(0,6,"EVM")]}
    ),
    (
        "Alternatively one can quote the precision for a fixed recall level such as the precision of the first K = 10 entities.",
        {"entities": [(54, 60, "EVM")]}
    ),
    (
        "The method had a precision of 66% when the recall was set to 10%; while low this is substantially more than rival variable-selection methods such as lasso and elastic net which were only slightly above chance.",
        {"entities": [(43, 49, "EVM")]}
    ),
# Alexa
    (
        "This week I read through a history of everything I've said to Alexa and it felt a little bit like reading an old diary.",
        {"entities": [(62, 67, "MLA")]}
    ),
    (
        "There are more than 100m Alexa-enabled devices in our homes.",
        {"entities": [(25, 30, "MLA")]}
    ),
    (
        "Amazon does a great job of giving you control over your privacy with Alexa.",
        {"entities": [(69, 74, "MLA")]}
    ),
    (
        "When you speak to Alexa a recording of what you asked Alexa is sent to Amazon.",
        {"entities": [(18, 23, "MLA")]}
    ),
    (
        "When you speak to Alexa a recording of what you asked Alexa is sent to Amazon.",
        {"entities": [(54, 59, "MLA")]}
    ),
    (
        "Alexa allows you to ask questions and make requests using just your voice.",
        {"entities": [(0,5,"MLA")]}
    ),
    (
        "you can ask Alexa a question, such as \"What is the weather today in New York?\"",
        {"entities": [(12,17,"MLA")]}
    ),
# NLP
    (
        "NLP has been considered a subdiscipline of Artificial Intelligence.",
        {"entities": [(0, 3, "MLA")]}
    ),
    (
        "Natural Language Processing (NLP) is a major area of artificial intelligence research which in its turn serves as a field of application and interaction of a number of other traditional AI areas.",
        {"entities": [(29, 32, "MLA")]}
    ),
    (
        "Natural Language Processing (NLP) is a major area of artificial intelligence research which in its turn serves as a field of application and interaction of a number of other traditional AI areas.",
        {"entities": [(0, 27, "MLA")]}
    ),
    (
        "The ontology is done using NLP technique where semantics relationships defined in WordNet.",
        {"entities": [(27, 30, "MLA")]}
    ),
    (
        "RU-EVAL is a biennial event organized in order to estimate the state of the art in Russian NLP resources methods and toolkits and to compare various methods and principles implemented for Russian.",
        {"entities": [(91, 94, "MLA")]}
    ),
    (
        "most commonly researched tasks in natural language processing.",
        {"entities": [(34,61,"MLA")]}
    ),
    (
        "In the early days, many language-processing systems were designed by hand-coding a set of rules",
        {"entities": [(24,43,"MLA")]}
    ),
# autonomous
    (
        "Two branches of the trend towards \"agents\" that are gaining currency are interface agents software that actively assists a user in operating an interactive interface and autonomous agents software that takes action without user intervention and operates concurrently.",
        {"entities": [(170, 180, "MLA")]}
    ),
    (
        "One category of research in Artificial Life is concerned with modeling and building so-called adaptive autonomous agents.",
        {"entities": [(103, 113, "MLA")]}
    ),
    (
        "This book deals with an important topic in distributed AI: the coordination of autonomous agents' activities.",
        {"entities": [(79, 89, "MLA")]}
    ),
    (
        "Developing autonomous or driver-assistance systems for complex urban traffic poses new algorithmic and system-architecture challenges.",
        {"entities": [(11, 21, "MLA")]}
    ),
    (
        "An autonomous floor-cleaning robot comprises a self-adjusting cleaning head subsystem that includes a dual-stage brush assembly having counter-rotating asymmetric brushes and an adjacent but independent vacuum assembly.",
        {"entities": [(3, 13, "MLA")]}
    ),
    (
        "AAFID was the first architecture that proposed the use of autonomous agents for doing intrusion detection.",
        {"entities": [(58, 68, "MLA")]}
    ),
    (
        "autonomous car",
        {"entities": [(0, 10, "MLA")]}
    ),
# bioinformatics
    (
        "This review article aims to provide an overview of the ways in which techniques from artificial intelligence can be usefully employed in bioinformatics both for modelling biological data and for making new discoveries.",
        {"entities": [(137, 151, "MLA")]}
    ),
    (
        "Artificial intelligence (AI) has increasingly gained attention in bioinformatics research and computational molecular biology.",
        {"entities": [(66, 80, "MLA")]}
    ),
    (
        "In this review the theory and main principles of the SVM approach are outlined and successful applications in traditional areas of bioinformatics research.",
        {"entities": [(131, 145, "MLA")]}
    ),
    (
        "Soft computing is make several latent in bioinformatics especially by generating low-cost low precision (approximate) good solutions.",
        {"entities": [(41, 55, "MLA")]}
    ),
    (
        "It has a wide spectrum of applications such as natural language processing search engines medical diagnosis bioinformatics and more.",
        {"entities": [(108, 122, "MLA")]}
    ),
    (
        "Clustering the rows and columns is known as biclustering or coclustering. This is widely used in bioinformatics, where the rows often represent genes and the columns represent conditions.",
        {"entities": [(97,111,"MLA")]}
    ),
    (
        "In recent years, the size and number of available biological datasets have skyrocketed, enabling bioinformatics researchers to make use of these machine learning systems.",
        {"entities": [(97,111,"MLA")]}
    ),
# pattern recognition
    (
        "This paper presents an electromyographic (EMG) pattern recognition method to identify motion commands for the control of a prosthetic arm by evidence accumulation based on artificial intelligence with multiple parameters.",
        {"entities": [(47, 66, "MLA")]}
    ),
    (
        "The techniques may be classified broadly into two categories—the conventional pattern recognition approach and the artificial intelligence (AI) based approach.",
        {"entities": [(78, 97, "MLA")]}
    ),
    (
        "We sought to test the hypothesis that a novel 2-dimensional echocardiographic image analysis system using artificial intelligence-learned pattern recognition can rapidly and reproducibly calculate ejection fraction (EF).",
        {"entities": [(138, 157, "MLA")]}
    ),
    (
        "This paper reports the use of a variety of pattern recognition techniques such as the learning machine and the Fisher discriminant.",
        {"entities": [(43, 62, "MLA")]}
    ),
    (
        "However my focus will not be on these types of pattern-recognition problems.",
        {"entities": [(47,66, "MLA")]}
    ),
    (
        "Pattern recognition is closely related to artificial intelligence and machine learning",
        {"entities": [(0,19,"MLA")]}
    ),
    (
        "Pattern recognition algorithms generally aim to provide a reasonable answer for all possible inputs and to perform \"most likely\" matching of the inputs, taking into account their statistical variation.",
        {"entities": [(0,19,"MLA")]}
    ),
# recomendations
#    (
#        "The avatar searches available information and makes recommendations to the user based on data.",
#        {"entities": [(52,67,"MLA")]}
#    ),
#    (
#        "Learning from GPS history data for collaborative recommendation.",
#        {"entities": [(49,63,"MLA")]}
#    ),
#    (
#        "A learning method was employed for Web pages recommendations and book recommendations in Mooney.",
#        {"entities": [(45,60,"MLA")]}
#    ),
# feature selection
    (
        "One common approach to tackling both of these problems is to perform feature selection to remove “irrelevant” features that do not help much with the classification problem.",
        {"entities": [(69, 86, "MLP")]}
    ),
    (
        "We introduced the topic of feature selection in Section 3.5.4 where we discussed methods for finding input variables which had high mutual information with the output.",
        {"entities": [(27, 44, "MLP")]}
    ),
    (
        "Feature selection in this context is equivalent to selecting a subset of the training examples which can help reduce overfitting and computational cost.",
        {"entities": [(0,17,"MLP")]}
    ),
    (
        "Note that the topic of feature selection and sparsity is currently one of the most active areas of machine learning/ statistics.",
        {"entities": [(23, 40, "MLP")]}
    ),
    (
        "To improve computational and statistical performance some feature selection was performed.",
        {"entities": [(58, 75, "MLP")]}
    ),
    (
        "We can create a challenging feature selection problem. In the experiments below we add 5 extra dummy variables.",
        {"entities": [(28, 45, "MLP")]}
    ),
    (
        "feature selection is the process of selecting a subset of relevant features for use in model construction",
        {"entities": [(0,17,"MLP")]}
    ),
# imputation
    (
        "The goal of imputation is to infer plausible values for the missing entries.",
        {"entities": [(12, 22, "MLP")]}
    ),
    (
        "An interesting example of an imputation-like task is known as image inpainting.",
        {"entities": [(29, 39, "MLP")]}
    ),
    (
        "Nevertheless the method can sometimes give reasonable results if there is not much missing data and it is a useful method for data imputation.",
        {"entities": [(131, 141, "MLP")]}
    ),
    (
        "As an example of this procedure in action let us reconsider the imputation problem from Section 4.3.2.3 which had N = 100 10-dimensional data cases with 50% missing data.",
        {"entities": [(64, 74, "MLP")]}
    ),
    (
        "Another interesting example of an imputation-like task is known as collaborative filtering.",
        {"entities": [(34, 44, "MLP")]}
    ),
    (
        "Imputation is the process of replacing missing data with substituted values",
        {"entities": [(0,10,"MLP")]}
    ),
    (
        "When imputed data is substituted for a data point, it is known as unit imputation",
        {"entities": [(71,81,"MLP")]}
    ),
# normalization
    (
        "The first term is just the normalization constant required to ensure the distribution sums to 1.",
        {"entities": [(27, 40, "MLP")]}
    ),
    (
        "The normalization constant only exists (and hence the pdf is only well defined) if ν > D − 1.",
        {"entities": [(4, 17, "MLP")]}
    ),
    (
        "So assuming the relevant normalization constants are tractable we have an easy way to compute the marginal likelihood.",
        {"entities": [(25, 38, "MLP")]}
    ),
    (
        "Since all the potentials are locally normalized since they are CPDs there is no need for a global normalization constant so Z = 1.",
        {"entities": [(98, 111, "MLP")]}
    ),
    (
        "Since all the potentials are locally normalized since they are CPDs there is no need for a global normalization constant so Z = 1.",
        {"entities": [(37, 47, "MLP")]}
    ),
    (
        "Hence satsifying normalization and local consistency is enough to define a valid distribution for any tree. Hence μ ∈ M(T ) as well.",
        {"entities": [(17, 30, "MLP")]}
    ),
    (
        "The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values.",
        {"entities": [(12,25,"MLP")]}
    ),
# pipeline
    (
        "This article reviews the evaluation and optimization of the preprocessing steps for bloodoxygenation-level-dependent (BOLD) functional magnetic resonance imaging (fMRI).",
        {"entities": []}
    ),
    (
        "However there is little consensus on the optimal choice of data preprocessing steps to minimize these effects.",
        {"entities": []}
    ),
    (
        "It has been established that the chosen preprocessing steps (or \"pipeline\") may significantly affect fMRI results.",
        {"entities": [(64, 72, "MLP")]}
    ),
    (
        "The automated analysis pipeline comprises data import normalization replica merging quality diagnostics and data export.",
        {"entities": [(23, 31, "MLP")]}
    ),
    (
        "The apparatus is formed as a pipeline having a translation and scaling section",
        {"entities": [(29, 37, "MLP")]}
    ),
    (
        "To address these challenges we developed an automated software pipeline called Rnnotator.",
        {"entities": [(63, 71, "MLP")]}
    ),
    (
        "pipelines consist of several steps to train a model",
        {"entities": [(0,9,"MLP")]}
    ),
# caffe
    (
        "We will use some Python code and a popular open source deep learning framework called Caffe to build the classifier.",
        {"entities": [(86,91,"MLS")]}
    ),
    (
        "Use tail -f model_1_train.log to view Caffe's progress.",
        {"entities": [(38,43,"MLS")]}
    ),
    (
        "The caffe \"tools/extra/parse_log.sh\" file requires a small change to use on OS X.",
        {"entities": [(4, 9, "MLS")]}
    ),
    (
        "I followed Caffe's tutorial on LeNet MNIST using GPU and it worked great.",
        {"entities": [(11, 16, "MLS")]}
    ),
    (
        "Alternatively Caffe has built in a function called iter_size.",
        {"entities": [(14, 19, "MLS")]}
    ),
    (
        "Caffe estimates the gradient (more accurately) weights are updated and the process continues.",
        {"entities": [(0, 5, "MLS")]}
    ),
    (
        "The feature iter_size is a Caffe function per se but you are correct that it is an option that you set in the solver protobuf file.",
        {"entities": [(27, 32, "MLS")]}
    ),
# keras
    (
        "Define your model using the easy to use interface of Keras.",
        {"entities": [(53, 58, "MLS")]}
    ),
    (
        "You can use the simple intuitive API provided by Keras to create your models.",
        {"entities": [(49, 54, "MLS")]}
    ),
    (
        "The Keras API is modular Pythonic and super easy to use.",
        {"entities": [(4, 9, "MLS")]}
    ),
    (
        "If you’re comfortable writing code using pure Keras go for it and keep doing it.",
        {"entities": [(46, 51, "MLS")]}
    ),
    (
        "Using Keras in deep learning allows for easy and fast prototyping as well as running seamlessly on CPU and GPU.",
        {"entities": [(6, 11, "MLS")]}
    ),
    (
        "The main advantages of Keras are described below.",
        {"entities": [(23, 28, "MLS")]}
    ),
    (
        "Preprocess input data for Keras",
        {"entities": [(27,31,"MLS")]}
    ),
# scikit learn
    (
        "One of the best known is Scikit-Learn a package that provides efficient versions of a large number of common algorithms.",
        {"entities": [(25, 37, "MLS")]}
    ),
    (
        "A benefit of this uniformity is that once you understand the basic use and syntax of Scikit-Learn for one type of model switching to a new model or algorithm is very straightforward.",
        {"entities": [(85, 97, "MLS")]}
    ),
    (
        "The best way to think about data within Scikit-Learn is in terms of tables of data.",
        {"entities": [(40, 52, "MLS")]}
    ),
    (
        "While some Scikit-Learn estimators do handle multiple target values in the form of a two-dimensional [n_samples n_targets] target array we will primarily be working with the common case of a one-dimensional target array.",
        {"entities": [(11, 23, "MLS")]}
    ),
    (
        "Many machine learning tasks can be expressed as sequences of more fundamental algorithms and Scikit-Learn makes use of this wherever possible.",
        {"entities": [(93, 105, "MLS")]}
    ),
    (
        "Scikit-learn (formerly scikits.learn) is a free software machine learning library for the Python",
        {"entities": [(0,12,"MLS")]}
    ),
    (
        "Scikit-learn plotting capabilities",
        {"entities": [(0,12,"MLS")]}
    ),
# spacy
    (
        "spaCy excels at large-scale information extraction tasks.",
        {"entities": [(0, 5, "MLS")]}
    ),
    (
        "Independent research in 2015 found spaCy to be the fastest in the world.",
        {"entities": [(35, 40, "MLS")]}
    ),
    (
        "With spaCy you can easily construct linguistically sophisticated statistical models for a variety of NLP problems.",
        {"entities": [(5, 10, "MLS")]}
    ),
    (
        "The new pretrain command teaches spaCy's CNN model to predict words based on their context.",
        {"entities": [(33, 38, "MLS")]}
    ),
    (
        "spaCy is an open-source software library for advanced natural language processing.",
        {"entities": [(0, 5, "MLS")]}
    ),
    (
        "spaCy excels at large-scale information extraction tasks.",
        {"entities": [(0,5,"MLS")]}
    ),
    (
        "spaCy comes with pretrained statistical models and word vectors, and currently supports tokenization for 50+ languages",
        {"entities": [(0,5,"MLS")]}
    ),
# tensorflow
    (
        "I’m not saying that you don’t need to understand a bit of TensorFlow for certain applications.",
        {"entities": [(58, 68, "MLS")]}
    ),
    (
        "TensorFlow is an end-to-end open source platform for machine learning. It’s a comprehensive and flexible ecosystem of tools libraries and other resources that provide workflows with high-level APIs.",
        {"entities": [(0, 10, "MLS")]}
    ),
    (
        "TensorFlow provides both high-level and low-level APIs.",
        {"entities": [(0, 10, "MLS")]}
    ),
    (
        "Tensorflow’s eager execution allows for immediate iteration along with intuitive debugging.",
        {"entities": [(0,10,"MLS")]}
    ),
    (
        "TensorFlow is designed for machine learning applications.",
        {"entities": [(0, 10, "MLS")]}
    ),
    (
        "TensorFlow is Google Brain's second-generation system",
        {"entities": [(0, 10, "MLS")]}
    ),
    (
        "TensorFlow is available on 64-bit Linux, macOS, Windows, and mobile computing platforms including Android and iOS",
        {"entities": [(0, 10, "MLS")]}
    ),
# Convolutional neural networks
    (
        "We trained a large deep convolutional neural network to classify the 1.3 million highresolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes.",
        {"entities": [(24,52,"NN")]}
    ),
    (
        "The ability to accurately represent sentences is central to language understanding. We describe a convolutional architecture dubbed the Dynamic Convolutional Neural Network (DCNN) that we adopt for the semantic modelling of sentences.",
        {"entities": [(136, 172, "NN")]}
    ),
    (
        "We propose two efficient approximations to standard convolutional neural networks: BinaryWeight-Networks and XNOR-Networks.",
        {"entities": [(52,81,"NN")]}
    ),
    (
        "Convolutional Neural Networks (CNNs) have been recently employed to solve problems from both the computer vision and medical image analysis fields.",
        {"entities": [(0, 29, "NN"),(31,34,"NN")]}
    ),
    (
        "We present a fast fully parameterizable GPU implementation of Convolutional Neural Network variants.",
        {"entities": [(62, 90, "NN")]}
    ),
    (
        "In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery.",
        {"entities": [(20,48,"NN"),(50,53,"NN"),(58,65,"NN")]}
    ),
    (
        "A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers",
        {"entities": [(2,28,"NN")]}
    ),
# boltzmann machine
    (
        "Part II addresses the problem of designing parallel annealing algorithms on the basis of Boltzmann machines.",
        {"entities": [(89, 106, "NN")]}
    ),
    (
        "I present a mean-field theory for Boltzmann machine learning derived by employing Thouless-Anderson-Palmer free energy formalism to a full extent.",
        {"entities": [(34, 51, "NN")]}
    ),
    (
        "We describe a model based on a Boltzmann machine with third-order connections that can learn how to accumulate information about a shape over several fixations.",
        {"entities": [(31, 48, "NN")]}
    ),
    (
        "Inspired by the success of Boltzmann machines based on classical Boltzmann distribution.",
        {"entities": [(27, 44, "NN")]}
    ),
    (
        "Paining a Boltzmann machine with hidden units is appropriately treated in information geometry using the information divergence and the technique of alternating minimization.",
        {"entities": [(10, 27, "NN")]}
    ),
    (
        "The main purpose of Boltzmann Machine is to optimize the solution of a problem.",
        {"entities": [(20,37,"NN")]}
    ),
    (
        "Boltzmann machines have fixed weights, hence there will be no training algorithm as we do not need to update the weights in the network.",
        {"entities": [(0,18,"NN")]}
    ),
# hopfield network
    (
        "The main application of Hopfield networks is as an associative memory or content addressable memory.",
        {"entities": [(24, 40, "NN")]}
    ),
    (
        "A Hopfield network (Hopfield 1982) is a fully connected Ising model with a symmetric weight matrix W = W T .",
        {"entities": [(2, 18, "NN")]}
    ),
    (
        "A large number of iterations and oscillations are those of the major concern in solving the economic load dispatch problem using the Hopfield neural network.",
        {"entities": [(133, 156, "NN")]}
    ),
    (
        "This paper formulates and studies a model of delayed impulsive Hopfield neural networks.",
        {"entities": [(63, 86, "NN")]}
    ),
    (
        "In this paper some novel criteria for the global robust stability of a class of interval Hopfield neural networks with constant delays are given.",
        {"entities": [(89, 112, "NN")]}
    ),
    (
        "A modified Hopfield neural network model for regularized image restoration is presented.",
        {"entities": [(11, 34, "NN")]}
    ),
    (
        "A Hopfield network is a form of recurrent artificial neural network popularized by John Hopfield in 1982",
        {"entities": [(2,18,"NN")]}
    ),
# perceptron
    (
        "We introduced a multilayer perceptron neural network (MLPNN) based classification model as a diagnostic decision support mechanism in the epilepsy treatment.",
        {"entities": [(27, 37, "NN")]}
    ),
    (
        "This study compares the performance of multilayer perceptron neural networks.",
        {"entities": [(50, 60, "NN")]}
    ),
    (
        "It is found to relax exponentially towards the perceptron of optimal stability using the concept of adaptive learning.",
        {"entities": [(47, 57, "NN")]}
    ),
    (
        "The perceptron: a probabilistic model for information storage and organization in the brain.",
        {"entities": [(4, 14, "NN")]}
    ),
    (
        "Perceptron training is widely applied in the natural language processing community for learning complex structured models.",
        {"entities": [(0,10,"NN")]}
    ),
    (
        "Rosenblatt made statements about the perceptron that caused a heated controversy among the fledgling AI community",
        {"entities": [(37,47,"NN")]}
    ),
    (
        "In the context of neural networks, a perceptron is an artificial neuron using the Heaviside step function as the activation function.",
        {"entities": [(37,47,"NN")]}
    ),
# restricted boltzmann machine
    (
        "Recent developments have demonstrated the capacity of restricted Boltzmann machines (RBM) to be powerful generative models able to extract useful features from input data or construct deep artificial neural networks.",
        {"entities": [(54, 82, "NN")]}
    ),
    (
        "The architecture is a continuous restricted Boltzmann machine with one step of Gibbs sampling to minimise contrastive divergence",
        {"entities": [(33, 61, "NN")]}
    ),
    (
        "We introduce the spike and slab Restricted Boltzmann Machine characterized by having both a real-valued vector the slab and a binary variable the spike associated with each unit in the hidden layer.",
        {"entities": [(32, 60, "NN")]}
    ),
    (
        "The restricted Boltzmann machine is a graphical model for binary random variables.",
        {"entities": [(4, 32, "NN")]}
    ),
    (
        "Restricted Boltzmann Machine (RBM) has shown great effectiveness in document modeling.",
        {"entities": [(0, 28, "NN")]}
    ),
    (
        "A restricted Boltzmann machine (RBM) is a generative stochastic artificial neural network that can learn a probability distribution over its set of inputs. ",
        {"entities": [(2,30,"NN"),(32,35,"NN")]}
    ),
    (
        "The standard type of RBM has binary-valued (Boolean/Bernoulli) hidden and visible units",
        {"entities": [(21,24,"NN")]}
    ),
# beam search
    (
        "In addition it uses a form of beam search to explore multiple paths through the lattice at once.",
        {"entities": [(30, 41, "OPM")]}
    ),
    (
        "A star search and beam search to quickly find an approximate MAP estimate.",
        {"entities": [(18, 29, "OPM")]}
    ),
    (
        "Mansinghka et al. 2007 discusses how to fit a DPMM online using particle filtering which is a like a stochastic version of beam search.",
        {"entities": [(123, 134, "OPM")]}
    ),
    (
        "The first use of a beam search was in the Harpy Speech Recognition System, CMU 1976.",
        {"entities": [(19,30,"OPM")]}
    ),
    (
        "Since local beam search often ends up on local maxima",
        {"entities": [(12,23,"OPM")]}
    ),
    (
        "A beam search is most often used to maintain tractability in large systems with insufficient amount of memory to store the entire search tree.",
        {"entities": [(2,13,"OPM")]}
    ),
    (
        "beam search returns the first solution found.",
        {"entities": [(0,11,"OPM")]}
    ),
# branch and bound
    (
        "A stochastic branch and bound method for solving stochastic global optimization problems is proposed.",
        {"entities": [(13, 29, "OPM")]}
    ),
    (
        "The idea to construct and solve entirely polyhedral-based relaxations in the context of branch-and-bound for global optimization was first proposed and analyzed by Taw- armalani and Sahinidis.",
        {"entities": [(88, 104, "OPM")]}
    ),
    (
        "The algorithm is of the branch-and-bound type and differs from previous interactive algorithms in several ways.",
        {"entities": [(24, 40, "OPM")]}
    ),
    (
        "This paper investigates the influence of the interval subdivision selection rule on the convergence of interval branch-and-bound algorithms for global optimization.",
        {"entities": [(112, 128, "OPM")]}
    ),
    (
        "A general branch-and-bound conceptual scheme for global optimization is presented that includes along with previous branch-and-bound approaches also grid-search techniques.",
        {"entities": [(10, 26, "OPM")]}
    ),
    (
        "Branch and bound (BB, B&B, or BnB) is an algorithm design paradigm for discrete and combinatorial optimization problems, as well as mathematical optimization.",
        {"entities": [(0,16,"OPM"),(18,20,"OPM"),(22,25,"OPM"),(30,33,"OPM")]}
    ),
    (
        "Branch-and-bound may also be a base of various heuristics.",
        {"entities": [(0,16,"OPM")]}
    ),
# gradient descent
    (
        "Perhaps the simplest algorithm for unconstrained optimization is gradient descent also known as steepest descent.",
        {"entities": [(65, 81, "OPM")]}
    ),
    (
        "The main issue in gradient descent is: how should we set the step size?",
        {"entities": [(18, 34, "OPM")]}
    ),
    (
        "This can be used inside a (stochastic) gradient descent procedure discussed in Section 8.5.2.",
        {"entities": [(39, 55, "OPM")]}
    ),
    (
        "As it stands WARP loss is still hard to optimize but it can be further approximated by Monte Carlo sampling and then optimized by gradient descent as described.",
        {"entities": [(130, 146, "OPM")]}
    ),
    (
        "It is straightforward to derive a gradient descent algorithm to fit this model; however it is rather slow.",
        {"entities": [(34, 50, "OPM")]}
    ),
    (
        "Then sketch how to use projected gradient descent to solve this problem.",
        {"entities": [(33, 49, "OPM")]}
    ),
    (
        "Since the Netflix data is so large (about 100 million observed entries) it is common to use stochastic gradient descent (Section 8.5.2) for this task.",
        {"entities": [(103, 119, "OPM")]}
    ),
# greedy search
    (
        "This is equivalent to performing a greedy search from the top of the lattice downwards.",
        {"entities": [(35, 48, "OPM")]}
    ),
    (
        "It is common to use greedy search to decide which variables to add.",
        {"entities": [(20, 33, "OPM")]}
    ),
    (
        "In practice greedy search techniques are used to find reasonable orderings (Kjaerulff 1990) although people have tried other heuristic methods for discrete optimization.",
        {"entities": [(12, 25, "OPM")]}
    ),
    (
        "An approximate method is to sample DAGs from the posterior and then to compute the fraction of times there is an s → t edge or path for each (s t) pair. The standard way to draw samples is to use the Metropolis Hastings algorithm (Section 24.3) where we use the same local proposal as we did in greedy search (Madigan and Raftery 1994).",
        {"entities": [(295, 308, "OPM")]}
    ),
    (
        "This precludes the kind of local search methods (both greedy search and MCMC sampling) we used to learn DAG structures.",
        {"entities": [(54, 67, "OPM")]}
    ),
    (
        "A greedy algorithm is any algorithm that follows the problem-solving heuristic of making the locally optimal choice at each stage with the intent of finding a global optimum",
        {"entities": [(2,18,"OPM")]}
    ),
    (
        "greedy strategy for the traveling salesman problem",
        {"entities": [(0,15,"OPM")]}
    ),
# CART
    (
        "This makes it clear that a CART model is just a an adaptive basis-function model.",
        {"entities": [(27, 31, "SML")]}
    ),
    (
        "CART models are popular for several reasons: they are easy to interpret 2  they can easily handle mixed discrete and continuous inputs.",
        {"entities": [(0, 4, "SML")]}
    ),
    (
        "However CART models also have some disadvantages.",
        {"entities": [(8, 12, "SML")]}
    ),
    (
        "“The HME approach is a promising competitor to CART trees”.",
        {"entities": [(47, 51, "SML")]}
    ),
    (
        "This weak learner can be any classification or regression algorithm but it is common to use a CART model.",
        {"entities": [(94, 98, "SML")]}
    ),
    (
        "The term Classification And Regression Tree (CART)",
        {"entities": [(9,43,"SML"),(45,49,"SML")]}
    ),
    (
        "An Introduction to Recursive Partitioning: Rationale, Application and Characteristics of Classification and Regression Trees, Bagging and Random Forests.",
        {"entities": [(89,124,"SML")]}
    ),
# SVM
    (
        "In fact many popular machine learning methods — such as support vector machines.",
        {"entities": [(56,63,"SML")]}
    ),
    (
        "Another very popular approach to creating a sparse kernel machine is to use a support vector machine or SVM.",
        {"entities": [(104, 107, "SML")]}
    ),
    (
        "SVM regression with C = 1/λ chosen by cross validation.",
        {"entities": [(0, 3, "SML")]}
    ),
    (
        "This combination of the kernel trick plus a modified loss function is known as a support vector machine or SVM.",
        {"entities": [(107, 110, "SML")]}
    ),
    (
        "It is possible to obtain sparse probabilistic multi-class kernel-based classifiers which work as well or better than SVMs.",
        {"entities": [(117, 120, "SML")]}
    ),
    (
        "In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis",
        {"entities": [(21,44,"SML"),(46,49,"SML"),(57,80,"SML")]}
    ),
    (
        "It is noteworthy that working in a higher-dimensional feature space increases the generalization error of support-vector machines, although given enough samples the algorithm still performs well",
        {"entities": [(106,129,"SML")]}
    ),
# decision tree
    (
        "Inputs in decision trees is to look for a series of ”backup” variables which can induce a similar partition to the chosen variable at any given split.",
        {"entities": [(10, 23, "SML")]}
    ),
    (
        "This can be thought of as a probabilistic decision tree of depth 2 since we recursively partition the space and apply a different expert to each partition.",
        {"entities": [(42, 55, "SML")]}
    ),
    (
        "By contrast in a standard decision tree predictions are made only based on the model in the corresponding leaf.",
        {"entities": [(26, 39, "SML")]}
    ),
    (
        "The standard heuristic for handling missing inputs in decision trees is to look for a series of ”backup” variables.",
        {"entities": [(54, 67, "SML")]}
    ),
    (
        "A simple decision tree for the data in Figure 1.1.",
        {"entities": [(9, 22, "SML")]}
    ),
    (
        "The decision tree can be linearized into decision rules",
        {"entities": [(4,17,"SML")]}
    ),
    (
        "Decision trees can also be seen as generative models of induction rules from empirical data",
        {"entities": [(0,14,"SML")]}
    ),
# naive bayes
    (
        "However even if the naive Bayes assumption is not true it oftenresults in classifiers that work well",
        {"entities": [(20, 31, "SML")]}
    ),
    (
        "We now discuss how to “train” a naive Bayes classifier.",
        {"entities": [(32, 43, "SML")]}
    ),
    (
        "If the sample size N is very small which model (naive Bayes or full) is likely to give lower test set error and why?",
        {"entities": [(48, 59, "SML")]}
    ),
    (
        "Hence in a naive Bayes classifier we can simply ignore missing features at test time.",
        {"entities": [(11, 22, "SML")]}
    ),
    (
        "So observing a root node separates its children (as in a naive Bayes classifier.",
        {"entities": [(57, 68, "SML")]}
    ),
    (
        "On the left we show a naive Bayes classifier that has been “unrolled” for D features.",
        {"entities": [(22, 33, "SML")]}
    ),
    (
        "naive Bayes classifiers can be trained very efficiently in a supervised learning setting",
        {"entities": [(0,11,"SML")]}
    ),
# random forest
    (
        "The technique known as random forests (Breiman 2001a) tries to decorrelate the base learners by learning trees based on a randomly chosen subset of input variables as well as a randomly chosen subset of data cases.",
        {"entities": [(23, 36, "SML")]}
    ),
    (
        "Note that the cost of these sampling-based Bayesian methods is comparable to the sampling-based random forest method.",
        {"entities": [(96, 109, "SML")]}
    ),
    (
        "The second best method was random forests invented by Breiman.",
        {"entities": [(27, 40, "SML")]}
    ),
    (
        "In second place are either random forests or boosted MLPs depending on the preprocessing.",
        {"entities": [(27, 40, "SML")]}
    ),
    (
        "Random forests or random decision forests are an ensemble learning method for classification, regression",
        {"entities": [(0,14,"SML"),(18,41,"SML")]}
    ),
    (
        "Random forests are a way of averaging multiple deep decision trees, trained on different parts of the same training set",
        {"entities": [(0,14,"SML")]}
    ),
    (
        "Similar to ordinary random forests, the number of randomly selected features to be considered at each node can be specified",
        {"entities": [(21,34,"SML")]}
    ),
# KNN
    (
        "A simple example of a non-parametric classifier is the K nearest neighbor (KNN) classifier.",
        {"entities": [(55,73,"USML"),(75, 78, "USML")]}
    ),
    (
        "A KNN classifier with K = 1 induces a Voronoi tessellation of the points.",
        {"entities": [(2, 5, "USML")]}
    ),
    (
        "The KNN classifier is simple and can work quite well provided it is given a good distance metric and has enough labeled training data.",
        {"entities": [(4, 7, "USML")]}
    ),
    (
        "However the main problem with KNN classifiers is that they do not work well with high dimensional inputs.",
        {"entities": [(30, 33, "USML")]}
    ),
    (
        "Choosing K for a KNN classifier is a special case of a more general problem known as model selection.",
        {"entities": [(17, 20, "USML")]}
    ),
    (
        "Often, the classification accuracy of k-NN can be improved significantly if the distance metric is learned with specialized algorithms",
        {"entities": [(38,42,"USML")]}
    ),
    (
        "The K-nearest neighbor classification performance can often be significantly improved through (supervised) metric learning",
        {"entities": [(4,37,"USML")]}
    ),
# clustering
    (
        "In astronomy the autoclass system (Cheeseman et al. 1988) discovered a new type of star based on clustering astrophysical measurements.",
        {"entities": [(97, 107, "USML")]}
    ),
    (
        "This procedure is called soft clustering and is identical to the computations performed when using a generative classifier.",
        {"entities": [(30, 40, "USML")]}
    ),
    (
        "We can represent the amount of uncertainty in the cluster assignment by using 1 − max k r ik . Assuming this is small it may be reasonable to compute a hard clustering using the MAP estimate.",
        {"entities": [(157, 167, "USML")]}
    ),
    (
        "As an example of clustering binary data consider a binarized version of the MNIST handwritten digit dataset.",
        {"entities": [(17, 27, "USML")]}
    ),
    (
        "After 20 iterations the algorithm has converged on a good clustering.",
        {"entities": [(58, 68, "USML")]}
    ),
    (
        "Clustering is an unsupervised task that may not yield a representation that is useful for prediction.",
        {"entities": [(0,10,"USML")]}
    ),
    (
        "Cluster analysis is for example used to identify groups of schools or students with similar properties.",
        {"entities": [(0,16,"USML")]}
    ),
# latent variable models
    (
        "However in general interpreting latent variable models is fraught with difficulties as we discuss in Section 12.1.3.",
        {"entities": [(32, 54, "USML")]}
    ),
    (
        "Now consider latent variable models of the form z i → x i ← θ.",
        {"entities": [(13, 35, "USML")]}
    ),
    (
        "A topic model is a latent variable model for text documents and other forms of discrete data.",
        {"entities": [(19, 40, "USML")]}
    ),
    (
        "If density estimation is our only goal it is worth considering whether it would be more appropriate to learn a latent variable model which can capture correlation between the visible variables via a set of latent common causes.",
        {"entities": [(111, 132, "USML")]}
    ),
    (
        "In this chapter we are concerned with latent variable models for discrete data such as bit vectors sequences of categorical variables count vectors graph structures relational data etc.",
        {"entities": [(38, 60, "USML")]}
    ),
    (
        "Many of the models we have looked at in this book have a simple two-layer architecture of the form z → y for unsupervised latent variable models or x → y for supervised models.",
        {"entities": [(122, 144, "USML")]}
    ),
    (
        "Latent Variable modeling can be a relevant tool for the optimization of analytical techniques",
        {"entities": [(0,24,"USML")]}
    ),
]
