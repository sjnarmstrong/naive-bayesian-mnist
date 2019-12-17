---
bibliography:
- 'reporttemplate.bib'
---

We have looked at the application of Bayesian approaches to linear
regression. We now turn to its application to classification. This is
the problem of assigning a class $c$ to a point in D-dimensional space
given by the coordinates $\mathbf{x}$. In this section we seek to
validate the use of a Nave Bayesian classifier to solve the problem of
optical character recognition. This is where an image of a character,
which is often hand-written, is used as an input. The algorithm must
then determine what that character is. The Nave Bayesian classifier is a
simple algorithm. However, this algorithm can provide effective
results.\
To test the effectiveness of this algorithm, we now turn to the MNIST
dataset[@lecun]. This consists of 60 000 handwritten digits of numbers,
between 0 and 9, and their corresponding labels. These can be used to
train the algorithm. It also provides a further 10 000 digits that can
be used to test the effectiveness of the developed algorithm. A benefit
of the MNIST dataset, is that all digits have been shifted to the centre
of the image and scaled to a set size. This reduces the amount of
preprocessing required to handle the digits, allowing more focus to be
put on the development of the classification algorithm. Sample digits
from the MNIST dataset are shown in figure \[fig:P2:MNIST\].

Selected features
-----------------

In order to apply the Nave Bayesian classifier to the MNIST dataset, we
first need features which can be extracted from the characters. We now
consider various features and provide motivation for their use.

![Sample of digits 0 to 9 from the MNIST
dataset[]{data-label="fig:P2:MNIST"}](Figs/P2/0.png "fig:"){width="9.00000%"}
![Sample of digits 0 to 9 from the MNIST
dataset[]{data-label="fig:P2:MNIST"}](Figs/P2/1.png "fig:"){width="9.00000%"}
![Sample of digits 0 to 9 from the MNIST
dataset[]{data-label="fig:P2:MNIST"}](Figs/P2/2.png "fig:"){width="9.00000%"}
![Sample of digits 0 to 9 from the MNIST
dataset[]{data-label="fig:P2:MNIST"}](Figs/P2/3.png "fig:"){width="9.00000%"}
![Sample of digits 0 to 9 from the MNIST
dataset[]{data-label="fig:P2:MNIST"}](Figs/P2/4.png "fig:"){width="9.00000%"}
![Sample of digits 0 to 9 from the MNIST
dataset[]{data-label="fig:P2:MNIST"}](Figs/P2/5.png "fig:"){width="9.00000%"}
![Sample of digits 0 to 9 from the MNIST
dataset[]{data-label="fig:P2:MNIST"}](Figs/P2/6.png "fig:"){width="9.00000%"}
![Sample of digits 0 to 9 from the MNIST
dataset[]{data-label="fig:P2:MNIST"}](Figs/P2/7.png "fig:"){width="9.00000%"}
![Sample of digits 0 to 9 from the MNIST
dataset[]{data-label="fig:P2:MNIST"}](Figs/P2/8.png "fig:"){width="9.00000%"}
![Sample of digits 0 to 9 from the MNIST
dataset[]{data-label="fig:P2:MNIST"}](Figs/P2/9.png "fig:"){width="9.00000%"}

### Ark Length

The ark length is defined as the length of the longest continuous set of
pixels, which have a value greater than a threshold. This feature was
considered in the hopes that each character has a different ark length.
This can be seen easily in the character “1” which has a very short
length when compared to other characters. It was decided to provide
various bins for this data rather than processing it as continuous
Gaussian data. This is because this descriptor will be multi-modal.
Providing bins for the data, we are better able to capture aspects of
this multi-modal distribution. We can analyse the effectiveness of this
descriptor by considering Kononenko’s information gain. This is defined
as follows: $$I_g(C|X)=\text{log}_bp(C|X)-\text{log}_bp(C)
\label{eqn:P2:IG}$$ Here we have defined the base of the logarithm as
$b$. It is common practice to use $b=2$, as this can denote the
effective number of bits gained in information. However, we will use
$b=M$, where $M=10$ is the number of classes. This choice of $b$,
provides an upper bound to the information gain as $I_g(c|x)\leq1$. It
is therefore easier to interoperate. An alternative view of this, is to
determine the information gain of a class given a predicted class $P$.
The information gain formula now becomes:
$$I_g(C|P)=\text{log}_{10}p(C|P)-\text{log}_{10}p(C)
\label{eqn:P2:IG10}$$ Using the training dataset to compute the
information gain yields the following result: $$\begin{aligned}[c]
&I_g(C=0|P=0)=0.4724\\
&I_g(C=1|P=1)=0.8630\\
&I_g(C=2|P=2)=0.3641\\
&I_g(C=3|P=3)=0.5294\\
&I_g(C=4|P=4)=0.3066
\end{aligned}
\quad\quad\quad\quad\quad\quad\quad\quad
\begin{aligned}[c]
&I_g(C=5|P=5)=0.7012\\
&I_g(C=6|P=6)=0.3529\\
&I_g(C=7|P=7)=0.4001\\
&I_g(C=8|P=8)=0.1622\\
&I_g(C=9|P=9)=0.4824
\end{aligned}$$

From $I_g(C=1|P=1)$, we can see that the ark length is good at
determining if a digit belongs to the class ’1’. It is also reasonable
at determining if a digit belongs to class ’5’.\
We can also plot the ark length versus the count of digits in each
class. This provides a visual on how well the descriptor splits the
data. This is shown in figure \[fig:P2:arkLenIG\] and confirms that this
descriptor is good at determining if a digit is a ’1’.\
The ark length alone does not provide excellent results but it does
split the data to a degree. Therefore, this feature is suitable to be
used in the Nave Bayesian classifier, provided it is used amongst other
descriptors.

![Bar graph of ark length and the corresponding count per
class[]{data-label="fig:P2:arkLenIG"}](Figs/P2/IGarkLen.png){width="65.00000%"}

### Area Enclosed

The next feature considered is the number of pixels enclosed by a
contour. This concept is explained with figure \[fig:P2:Enclosed0\],
where the area enclosed is coloured in red. The reason for choosing this
descriptor is to try and distinguish between numbers enclosing a space
and those that don’t. We also hope to distinguish various characters by
how much space they enclose. This can be seen when considering a ’0’ and
a ’6’. More often than not, a ’0’ encloses more total space than a ’6’.\

![Image depicting the number of pixels
enclosed[]{data-label="fig:P2:Enclosed0"}](Figs/P2/0filled.png){width="20.00000%"}

The information gain for this feature is given below:
$$\begin{aligned}[c]
&I_g(C=0|P=0)=0.9143\\
&I_g(C=1|P=1)=0.2468\\
&I_g(C=2|P=2)=0.2845\\
&I_g(C=3|P=3)=-0.0043\\
&I_g(C=4|P=4)=-0.0336
\end{aligned}
\quad\quad\quad\quad\quad\quad\quad\quad
\begin{aligned}[c]
&I_g(C=5|P=5)=0.0491\\
&I_g(C=6|P=6)=0.5271\\
&I_g(C=7|P=7)=0.1131\\
&I_g(C=8|P=8)=0.5912\\
&I_g(C=9|P=9)=0.5291
\end{aligned}$$ This feature does exactly as expected and provides a
large information gain to characters which encircle pixels This feature
assigns the same value to characters that don’t enclose a space and
hence don’t provide much information. Figure \[fig:P2:IGEnclosed\]
provides a visual depiction on how this feature splits the data. The bin
associated with 0 pixels has been removed as a large portion of the data
was contained in this bin. Due to this descriptors ability to
differentiate between digits which encircle an area, it is a suitable
feature to use in the Nave Bayesian classifier.

![Bar graph of area enclosed and the corresponding count per
class[]{data-label="fig:P2:IGEnclosed"}](Figs/P2/IGareaEnclosed.png){width="70.00000%"}

### Number of Contours

The number of contours is defined as the number of continuous edges of
the digit. This concept is shown in figure \[fig:P2:8Cont\], where each
contour is highlighted in orange. The total number of contours in this
image is 3. Small contours are removed from this count to avoid counting
fullstops which are present in some of the images. This feature is
included to attempt to distinguish ’8’ better from other digits. This is
useful as many descriptors struggle to differentiate between ’3’ and
’8’.\

![Image highlighting all contours of the digit
8[]{data-label="fig:P2:8Cont"}](Figs/P2/8Cont.png){width="20.00000%"}

The information gain of this algorithm is given in equation
\[eqn:P2:IGNOC\], and a histogram containing the number of classes, for
each value of this descriptor, is given in feature \[fig:P2:IGCont\].
Both of these show that this feature is indeed able to identify an ’8’
to some degree. Therefore, this feature will provide useful information
to the Nave Bayesian classifier. $$\begin{aligned}[c]
&I_g(C=0|P=0)=0.4631\\
&I_g(C=1|P=1)=0.2472\\
&I_g(C=2|P=2)=-0.0135\\
&I_g(C=3|P=3)=-0.0043\\
&I_g(C=4|P=4)=-0.0078
\end{aligned}
\quad\quad\quad\quad\quad\quad\quad\quad
\begin{aligned}[c]
&I_g(C=5|P=5)=0.0491\\
&I_g(C=6|P=6)=0.0184\\
&I_g(C=7|P=7)=-0.0119\\
&I_g(C=8|P=8)=0.7254\\
&I_g(C=9|P=9)=-0.0039
\end{aligned}
\label{eqn:P2:IGNOC}$$

![Bar graph of the number of contours and the corresponding count per
class[]{data-label="fig:P2:IGCont"}](Figs/P2/IGnumValidContours.png){width="70.00000%"}

### Histogram of Oriented Gradients

This feature moves a window over the image and determines the magnitude
and direction of the gradients in that window. It then builds a
histogram containing bins for various gradient directions. This feature
is used in various human detection algorithms. This descriptor was used
in [@ebrahimzadeh2014efficient] for OCR. This algorithm was tested on
the MNIST and obtained an accuracy of $97.25\%$. Due to this, this
feature is of great interest.\
We are able to use this feature by considering each bar as a separate
feature which can be used by the algorithm. One can also adjust the
window size in order to change the granularity of the search. Smaller
windows tend to provide a better accuracy up to a point but require more
time to compute and more space to store.\
The information gain of all the features produced by the HOG algorithm
is given in equation \[eqn:P2:IGHOG\]. Samples of some of the features
produced by the algorithm are also provided in figure \[fig:P2:IGHOG\].
This feature yields excellent results and could even be used by itself.
$$\begin{aligned}[c]
&I_g(C=0|P=0)=0.9726\\
&I_g(C=1|P=1)=0.9185\\
&I_g(C=2|P=2)=0.9057\\
&I_g(C=3|P=3)=0.9295\\
&I_g(C=4|P=4)=0.9652
\end{aligned}
\quad\quad\quad\quad\quad\quad\quad\quad
\begin{aligned}[c]
&I_g(C=5|P=5)=0.992\\
&I_g(C=6|P=6)=0.9926\\
&I_g(C=7|P=7)=0.948\\
&I_g(C=8|P=8)=0.9046\\
&I_g(C=9|P=9)=0.9096
\end{aligned}
\label{eqn:P2:IGHOG}$$

![Plot of randomly selected HOG descriptors and their likelihood
functions for each
class[]{data-label="fig:P2:IGHOG"}](Figs/P2/HOG0.png "fig:"){width="49.00000%"}
![Plot of randomly selected HOG descriptors and their likelihood
functions for each
class[]{data-label="fig:P2:IGHOG"}](Figs/P2/HOG16.png "fig:"){width="49.00000%"}
![Plot of randomly selected HOG descriptors and their likelihood
functions for each
class[]{data-label="fig:P2:IGHOG"}](Figs/P2/HOG21.png "fig:"){width="49.00000%"}
![Plot of randomly selected HOG descriptors and their likelihood
functions for each
class[]{data-label="fig:P2:IGHOG"}](Figs/P2/HOG24.png "fig:"){width="49.00000%"}

### Image Moments

Image moments provide a numerical value for how the image intensities
are distributed. The following equation, given in [@1057692], is used to
calculate image moments:
$$\mu_{pq}=\sum_{p}^{m}\sum_{q}^{n}\begin{pmatrix}
    p \\
    m \\
    \end{pmatrix}\begin{pmatrix}
    q \\
    n \\
    \end{pmatrix}(-\bar{x})^{p-m}(-\bar{y})^{q-n}M_{mn}$$ Where $M_{ij}$
is given in equation \[eqn:P2:Mij\] and $I(x,y)$ is the pixel intensity
at position $(x,y)$. $$M_{ij}=\sum_{x}\sum_{y}I(x,y)
\label{eqn:P2:Mij}$$

The information gain of this feature is given in equation
\[eqn:P2:IGGrad\]. Figures \[fig:P2:IGGrad\] and \[fig:P2:IGGrad2\]
provides the histograms containing the likelihood of each moment
considered. This descriptor also provides excellent results in splitting
the data.

$$\begin{aligned}[c]
&I_g(C=0|P=0)=0.8710\\
&I_g(C=1|P=1)=0.7913\\
&I_g(C=2|P=2)=0.7690\\
&I_g(C=3|P=3)=0.8472\\
&I_g(C=4|P=4)=0.8195
\end{aligned}
\quad\quad\quad\quad\quad\quad\quad\quad
\begin{aligned}[c]
&I_g(C=5|P=5)=0.7489\\
&I_g(C=6|P=6)=0.8500\\
&I_g(C=7|P=7)=0.8716\\
&I_g(C=8|P=8)=0.6934\\
&I_g(C=9|P=9)=0.6988
\end{aligned}
\label{eqn:P2:IGGrad}$$

![Plot of various moments and their likelihood function for each
class[]{data-label="fig:P2:IGGrad"}](Figs/P2/Momentmu00.png "fig:"){width="49.00000%"}
![Plot of various moments and their likelihood function for each
class[]{data-label="fig:P2:IGGrad"}](Figs/P2/Momentmu02.png "fig:"){width="49.00000%"}
![Plot of various moments and their likelihood function for each
class[]{data-label="fig:P2:IGGrad"}](Figs/P2/Momentmu03.png "fig:"){width="49.00000%"}
![Plot of various moments and their likelihood function for each
class[]{data-label="fig:P2:IGGrad"}](Figs/P2/Momentmu11.png "fig:"){width="49.00000%"}

![Plot of various moments and their likelihood function for each
class[]{data-label="fig:P2:IGGrad2"}](Figs/P2/Momentmu12.png "fig:"){width="49.00000%"}
![Plot of various moments and their likelihood function for each
class[]{data-label="fig:P2:IGGrad2"}](Figs/P2/Momentmu20.png "fig:"){width="49.00000%"}
![Plot of various moments and their likelihood function for each
class[]{data-label="fig:P2:IGGrad2"}](Figs/P2/Momentmu21.png "fig:"){width="49.00000%"}
![Plot of various moments and their likelihood function for each
class[]{data-label="fig:P2:IGGrad2"}](Figs/P2/Momentmu30.png "fig:"){width="49.00000%"}

Theoretical Analysis {#sec:P2:theorAnal}
--------------------

In order to classify he data, we look to derive the probability
$p(c|x)$. This is easily determined using Bayes rule as follows:
$$p(c|x)=\frac{p(x|c)p(c)}{p(x)}$$ If we now assume that the prior
probability $p(c)$ is relatively flat over all the classes, we can
simplify this to: $$p(c|x)\propto p(x|c)$$

This result can now be extended to multiple features as follows:
$$p(c|x_1,...,x_M)\propto p(x_1,...,x_M|c)
\label{eqn:cGxpxGc}$$ This can be further simplified, as shown in figure
\[eqn:P3:indPxGx\], if we assume that each feature is independent. This
requires that features should not depend on similar data, which is not
always practically possible. However, if the dependence between features
are kept minimal, the mutual independence assumption provides a decent
approximation to the true distribution.
$$p(x_1,...,x_M|c)\propto p(x_1|c)p(x_2|c)...p(x_M|c)
\label{eqn:P3:indPxGx}$$ It is important to note that this algorithm
only works in one dimension at a time. Better results could be achieved
if the relations between multiple features where considered together in
a D-dimensional space. However, this would require more computation
power and more complex mechanisms. Hence there is a trade-off between
accuracy and complexity. However, the accuracy gain can often be
minimal.\
We now consider the evaluation of $p(x|c)$. There are 2 important cases
to consider when determining this probability from the training data.
These two cases are when $x$ is discrete and when $x$ is continuous. We
first consider the case where $x$ is discrete and can take on one of K
values, formally denoted $x_i\in \lbrace v_1, v_2,..., v_k\rbrace$. The
$p(x|c)$ can be simply given as:
$$p(x=v_i|c=c_j)=\frac{\eta_{i,j}}{\eta_{j}}$$ Where $\eta_{i,j}$ is the
count of the data observed having value $x=v_i$ and class $c=c_j$. We
have also defined $\eta_{j}$ as the count of data observed belonging to
class $c_j$.\
We now turn our attention to the case of $x$ as a continuous variable.
In order to evaluate $p(x|c)$, we need to make an assumption on the
expected form of this probability distribution. We assume that the data
takes on the form of a Gaussian distribution. This distribution appears
often due to the central limit theory. However, various features can
have a multi-modal distribution. Therefore, it is possible to miss some
detail in the distribution by making this assumption. It still produces
excellent results when put to the test. With the assumption on the form
of $p(x|c)$, we can show that:
$$p(x|c)=\alpha \mathcal{N}(x|\mu_c,\sigma_c^2)$$ Where we can use the
following equations to estimate $\mu_c$ and $\sigma_c^2$ from the
training data within each class: $$\begin{aligned}
&\mu_c=\frac{1}{N_c}\sum_{i=1}^{N_c}x_c\label{eqn:P2:mu}\\
&\sigma_c^2=\frac{1}{N_c-1}\sum_{i=1}^{N_c}(x_c-\mu_c)^2\label{eqn:P2:sigma}\end{aligned}$$
We can the normalise $p(x|c)$ over all the classes to calculate the
value of $\alpha$. It is important to notice the minus 1 in the
denominator of equation \[eqn:P2:sigma\], as this accounts for the
degree of freedom used when fitting the mean to the data.

Algorithmic Development
-----------------------

There are 2 main stages in the Nave Bayesian classifier. Firstly, the
algorithm needs to be trained. We then need to make predictions with the
trained models. The implementation of these are considered separately
below.

### Training

Discrete variables can be trained with the use of a matrix which has the
classes across the columns and the $x$ across the rows. Each new
observation in the training set then just increments the counter in the
relevant row and column. One important adjustment is to set the initial
count of this matrix to 1. This is done to avoid a ’0’ probability
occurring which can greatly influence the overall probability. The
initialisation and training for discrete variable are given in
algorithms \[alg:DI\] and \[alg:DT\].

$\textit{output labels} \gets \text{N dimentional vector of output labels}$
$\textit{x labels} \gets \text{M dimentional vector of posibile x values}$
$\textit{Probability Matrix} \gets \text{An N by M matrix initialised to all 1's}$

$\textit{i} \gets \text{Position of c in output labels}$
$\textit{j} \gets \text{Position of x in x labels}$
$\textit{Probability Matrix}[i][j] \gets \text{Probability Matrix}[i][j] + 1$

When training continuous variables, we need to consider all the data as
a whole to calculate the mean and variance. Due to this, the training is
split into two parts. The first part collects all the data associated
with each class and the second part calculated the mean and average
within each class. Algorithms \[alg:CI\], \[alg:CO\] and \[alg:CT\] show
how this could be implemented.

$\textit{output labels} \gets \text{N dimentional vector of output labels}$
$\textit{x labels} \gets \text{M dimentional vector of posibile x values}$
$\textit{Observed lists} \gets \text{A list structure containing N empty arrays}$

$\textit{i} \gets \text{Position of c in output labels}$
$\text{Append}(\textit{Observed lists}[i],x)$

$\mu \gets \text{Mean, given by \ref{eqn:P2:mu}, for each array in }\textit{Observed lists}$
$\sigma^2 \gets \text{Variance, given by \ref{eqn:P2:sigma}, for each array in }\textit{Observed lists}$

### Predictions

We now show how the trained classifiers can be used to make predictions.
To do this we first need to calculate the $p(x|c)$ for both discrete and
continuous variables. Algorithms \[alg:DP\] and \[alg:CP\] provide
possible pseudo code for achieving this. One we know the probability for
each input $x$, we can simply multiply all of these together to get the
total probability $p(x_1,..x_M|c)$. Due to equation \[eqn:cGxpxGc\], we
can now assign the point given by $\mathbf{x}$ by the class that gives
the maximum value for $p(x_1,..x_M|c)$.

$\textit{j} \gets \text{Position of x in }\textit{x labels}$

$\eta_{i} \gets 0$ $\textit{probability} \gets \text{A new empty array}$
*loop over classes c*:
$\textit{i} \gets \text{Position of c in output labels}$
$\eta_{i,j} \gets \textit{Probability Matrix}[i][j]$
$\eta_{i} \gets \eta_{i}+\eta_{i,j}$
$\text{Append}(\textit{probability},\eta_{i,j})$
$\textit{probability}/\eta_{i}$

$P_t \gets 0$ $\textit{probability} \gets \text{A new empty array}$
*loop over classes c*:
$\textit{i} \gets \text{Position of c in output labels}$
$Px_c \gets \mathcal{N}(x,\mu_i|\sigma_i^2)$ $P_t \gets P_t+Px_c$
$\text{Append}(\textit{probability},Px_c)$ $\textit{probability}/P_t$

Experimental protocol
---------------------

To test this algorithm, we will use MNIST dataset. Features will be
generated for both the test set and the training set which contains 10
000 digits and 60 000 digits respectively. The generated features of the
training set and their associated labels are then used to train the
classifier. Thereafter, we will use the training set features to
classify each digit. It is important that we use a separate dataset for
testing and training as this allows us to test for over-fitting. In this
section, the tests performed will be described.

### Experiment 1: Confusion matrix and overall accuracy

**Aim**\
This experiment aims to acquire the mean accuracy of the system and also
provide a confusion matrix. The confusion matrix will be used to provide
insight into what the algorithm struggles to classify. This could be
used to make later improvements.\
**Expectations**\
One could expect to see many misclassification of the digit ’7’ and ’9’
as these two digits are very similar in structure. There is also very
little structural difference between an ’8’ and a ’0’.\
**Procedure**\
The following is performed in order to obtain the results:

1.  A 10$\times$10 matrix is initialised to 0. This is used as the
    confusion matrix.

2.  An item containing the features of a single digit is read from a
    file.

3.  This digit is classified setting the most likely class as the
    predicted class.

4.  The actual class is also loaded from the file.

5.  The matrix value at the index provided by the actual label and the
    predicted label is incremented by 1.

6.  This process is repeated for all the test digits provided by the
    MNIST dataset.

7.  The mean accuracy is then calculated as
    $100\times\text{Trace}(\textit{confusion matrix})/\text{Sum}(\textit{confusion matrix})$

### Experiment 2: Sub-Sampling

**Aim**\
This experiment aims to provide an indication on how accurate this
classifier would work on various other datasets. This is achieved by
sub-sampling the test dataset and testing the variance of the
sub-samples.\
**Expectations**\
One would expect that the classifier would yield similar results for
similar datasets.\
**Procedure**\
The following is performed in order to obtain the results:

1.  Two empty lists are initialised

2.  An item containing the features of a single digit is read from a
    file.

3.  The probabilities associated with this digit belonging to the
    various classes are determined as described in section
    \[sec:P2:theorAnal\].

4.  A ’1’ is appended to an array if the correct classification was
    made, else a ’0’ is appended.

5.  We also obtain the classifies probability of the actual label being
    correct. This is done by indexing the probability array with the
    actual label. This is also appended to a list

6.  This process is repeated for all the test digits provided by the
    MNIST dataset.

7.  We now take 1000 random test samples from the first array and count
    the total number of correctly classified digits (1’s). It is
    important to note that the 1000 samples are taken from a uniform
    distribution with no repetition in digits.

8.  This number is then divided by 10 and appended to a list to get the
    accuracy in percentage.

9.  We also take 1000 random samples out of the second list in a similar
    fashion.

10. The sum of these probabilities are calculated and divided by 10 to
    get how sure the classifier is of the correct decision, on average,
    in percentage.

11. This sampling process is repeated 100 000 times.

12. The respective means, medians, quartiles, minimums and maximums can
    be calculated. These provide insight into how the accuracy varies
    with a variation in the test dataset.

Results
-------

### Experiment 1: Confusion matrix and overall accuracy

The confusion matrix for this classifier is given in table \[tbl:Conf\]
and the overall accuracy is determined to be $89.97\%$.

  -- --- ----- ------ ----- ----- ----- ----- ----- ----- ----- -----
                                                                
     0   1     2      3     4     5     6     7     8     9     
     0   915   2      13    8     0     4     15    0     20    3
     1   0     1075   19    2     13    1     0     5     19    1
     2   9     1      956   29    3     2     0     4     27    1
     3   0     2      26    925   2     23    1     7     16    8
     4   0     7      27    0     892   2     10    8     5     31
     5   7     0      7     50    3     777   10    2     32    4
     6   10    10     7     1     9     26    881   0     14    0
     7   0     6      35    10    32    0     0     842   10    93
     8   30    9      23    26    9     16    2     11    828   20
     9   9     9      17    10    8     8     0     26    16    906
  -- --- ----- ------ ----- ----- ----- ----- ----- ----- ----- -----

  : Confusion matrix for the Nave Bayesian
  classifier[]{data-label="tbl:Conf"}

### Experiment 2: Sub-Sampling

Figure \[IMG:P2:BoxAndWhisacar\] shows the box and whisker diagram for
the randomly sampled data. In this experiment, we obtain a mean accuracy
of $89.97\%$ and a mean certainty of $89.92\%$.

\[ ytick=[1,2]{}, yticklabels=[Certainty, Accuracy]{}, \] +\[ boxplot
prepared=[ median=89.93, upper quartile=90.53, lower quartile=89.33,
upper whisker=93.57, lower whisker=85.97 ]{}, \] coordinates ; +\[
boxplot prepared=[ median=90.0, upper quartile=90.6, lower
quartile=89.4, upper whisker=93.9, lower whisker=85.3 ]{}, \]
coordinates ;

Analysis
--------

The Nave Bayesian classifier was able to take a set of input variables
and assign classes to a set of target variables with a high accuracy.
This is very impressive when considering the simplicity of the
algorithm. However, it would be possible to achieve a higher accuracy if
we where able to select features which where completely independent.
This would be equivalent to warping the D-Dimensional space to an axis
which splits the data the most.\
It is interesting to see how each feature contributes to the
effectiveness of the algorithm. When analysing the information gain, we
can see that the ark length is good at separating 1’s and 5’s out from
other digits. The detection of the digit ’1’ can be explain by its
relatively short ark length. The digit ’5’ on the other hand is due to
its consistent ark length when compared to other digits like ’4’. The
next feature that can be considered is the area enclosed. This feature
is best at detecting 0’s due to the large area enclosed in the centre.
Using he number of contours as a feature, we could detect 8’s really
well. This is because the figure ’8’ usually contains 3 distinct
contours. These features provided minimal information gain when compared
to the HOG descriptor and the moments of the image. This is due to the
amount of information that can be captured by these features.\
Due to the confusion matrix, it is possible to determine where the
algorithm excelled and where improvements could be made. A large portion
of the invalid predictions are due to 7’s getting detected as 9’s and
visa versa. This amounts to a total of 119 misclassified images, which
is equivalent to $1.19\%$. When considering the similarity of a ’7’ and
a ’9’, this result could be expected. This effect is magnified by the
various handwriting styles, where a ’7’ could be drawn with
imperfections which could make it visually similar to a ’9’. Looking
further into the confusion matrix, a better accuracy could be achieved
if more features were tailored to detecting 8’s and 9’s, as most of the
misclassification’s occur in these columns.\
In experiment 2, the consistency of the classifier is tested by using
smaller random samples of the bigger test set. Assuming that the larger
test set contains enough data-points to represent all possible data
which could be provided to the classifier, we can assume that the
variance of the algorithm within the dataset, will approximate the
variance of the algorithm in a real-world environment. Therefore, figure
\[IMG:P2:BoxAndWhisacar\] shows us that the algorithm will most likely
have an accuracy of at least $85\%$ for most real world data. However,
with the MNIST dataset, the classifier obtained an average accuracy
$89.97\%$. This is poor when compared to recent algorithms, which can
achieve error rates of only $0.02\%$[@lecun]. However, these modern
algorithms can take advantage of multiple layers of abstraction,
creating linear separations to complex relationships between the data.

Summary
=======

In this paper, we have seen how probability theory can be used to
quantify uncertainty. We have also seen how this theorem is applicable
to many real-life scenarios and how they can be used to predict the
likelihood of various events occurring. We also see how ones intuition
on probabilities can often be incorrect.\
We continue to explore the application of probability theory to the
problem of regression. This is compared to a frequentest approach, where
the advantage of a probabilistic approach become apparent. One can see
that a probabilistic approach avoids the problem of over fitting, which
is present when using a frequentest approach. This can be seen when
comparing figures \[fig:E3:Or9:LSQ\] and \[fig:E3:Or9:Bays\]. One can
also see that, a probabilistic approach, allows one to determine the
level of certainty that the data will be contained within a certain
range. It is interesting to not that the certainty decreases when there
is a lack of data around a given range of input variables. This can be
seen in figure \[fig:E3:Or9:Bays:RandRem\], where observations have been
removed. This creates a distribution which is more broad around the area
of the missing data. One would expect this to occur as the algorithm has
no information about the shape of the curve in the proximity of these
input variables. The final advantage of using a probabilistic approach
lies in model selection. Given a list of possible models, a
probabilistic approach allows one to select the most probable model
based on the training data alone. This is not possible using a
frequentest approach and would require a separate testing dataset to
avoid over fitting. This is a large waste of data which could be used
for further training.\
Lastly, we look at how a probabilist approach could be taken to data
classification. A Nave Bayesian classifier was developed to solve the
problem of optical character recognition (OCR). This classifier needs
either discrete or continuous data from an image. In order to achieve
this, various feature extraction methods where explored and their use
validated. The classification algorithm could then be tested on the
MNIST dataset[@lecun], where it achieved an average accuracy of
$89.97\%$.

Conclusion
==========

Probability theory provides a powerful framework, whereby predictions
can be made from previous observations. Therefore, this is an essential
tool in artificial intelligence and forms the basis for the majority of
machine learning algorithms. It is undeniable that probabilistic
approaches provide significant benefits when used for regression and
classification. We see that problems such as over-fitting, model
selection and certainty estimation can be solved with probabilistic
approaches to linear regression. Finally, we see that a probabilistic
approach to the problem of classification, can provide a simplistic but
effective approach to a very complex problem.
