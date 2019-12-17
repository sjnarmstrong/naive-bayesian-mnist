---
bibliography:
- 'reporttemplate.bib'
---

We have looked at the application of Bayesian approaches to linear
regression. We now turn to its application to classification. This is
the problem of assigning a class <img src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg" align=middle width=7.113876pt height=14.15535pt/> to a point in D-dimensional space
given by the coordinates <img src="svgs/b0ea07dc5c00127344a1cad40467b8de.svg" align=middle width=9.97722pt height=14.61207pt/>. In this section we seek to
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
as follows: <p align="center"><img src="svgs/7f8ee93319740b1d7486242488ab964b.svg" align=middle width=244.7181pt height=17.03196pt/></p> Here we have defined the base of the logarithm as
<img src="svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg" align=middle width=7.0548555pt height=22.83138pt/>. It is common practice to use <img src="svgs/0a37c25b1ae998753ecf173c409467b7.svg" align=middle width=37.19166pt height=22.83138pt/>, as this can denote the
effective number of bits gained in information. However, we will use
<img src="svgs/bfbdd08513df029e9e1f0fe16883f3b0.svg" align=middle width=46.71216pt height=22.83138pt/>, where <img src="svgs/2914927ad3405cbe6353426edae81b59.svg" align=middle width=56.095875pt height=22.46574pt/> is the number of classes. This choice of <img src="svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg" align=middle width=7.0548555pt height=22.83138pt/>,
provides an upper bound to the information gain as <img src="svgs/d7e79938a1ce1049197192e582433070.svg" align=middle width=78.87132pt height=24.6576pt/>. It
is therefore easier to interoperate. An alternative view of this, is to
determine the information gain of a class given a predicted class <img src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg" align=middle width=12.836835pt height=22.46574pt/>.
The information gain formula now becomes:
<p align="center"><img src="svgs/6cfa17024e464f2a91082f579a10a752.svg" align=middle width=255.22365pt height=17.03196pt/></p> Using the training dataset to compute the
information gain yields the following result: <p align="center"><img src="svgs/6215c69ce3f5aa93aa45c066e25311a3.svg" align=middle width=503.1873pt height=115.66203pt/></p>

From <img src="svgs/ff6e214af043c91e45f02c08fd836afc.svg" align=middle width=118.26078pt height=24.6576pt/>, we can see that the ark length is good at
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
<p align="center"><img src="svgs/6746ec89ef5633ab592aedfc31069317.svg" align=middle width=515.97315pt height=115.66203pt/></p> This feature does exactly as expected and provides a
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
to the Nave Bayesian classifier. <p align="center"><img src="svgs/f505f4caaac2a9acc04c5efad824d2ce.svg" align=middle width=528.759pt height=115.66203pt/></p>

![Bar graph of the number of contours and the corresponding count per
class[]{data-label="fig:P2:IGCont"}](Figs/P2/IGnumValidContours.png){width="70.00000%"}

### Histogram of Oriented Gradients

This feature moves a window over the image and determines the magnitude
and direction of the gradients in that window. It then builds a
histogram containing bins for various gradient directions. This feature
is used in various human detection algorithms. This descriptor was used
in [@ebrahimzadeh2014efficient] for OCR. This algorithm was tested on
the MNIST and obtained an accuracy of <img src="svgs/2ba4bfd1c850ee20934a36e70ddf7bdb.svg" align=middle width=51.14175pt height=24.6576pt/>. Due to this, this
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
<p align="center"><img src="svgs/89bdf1320374ba15c81bd6805d52e24c.svg" align=middle width=503.1873pt height=115.66203pt/></p>

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
<p align="center"><img src="svgs/d5970412846325bc9c8fe0e4dd191a03.svg" align=middle width=337.12965pt height=46.644015pt/></p> Where $M_{ij}$
is given in equation \[eqn:P2:Mij\] and <img src="svgs/f2777ca1da2df7338157e0e1609cfe0b.svg" align=middle width=46.651605pt height=24.6576pt/> is the pixel intensity
at position <img src="svgs/7392a8cd69b275fa1798ef94c839d2e0.svg" align=middle width=38.135625pt height=24.6576pt/>. <p align="center"><img src="svgs/acec83cdc94b0bd17bb9d89b8f2e3f76.svg" align=middle width=149.061825pt height=38.4021pt/></p>

The information gain of this feature is given in equation
\[eqn:P2:IGGrad\]. Figures \[fig:P2:IGGrad\] and \[fig:P2:IGGrad2\]
provides the histograms containing the likelihood of each moment
considered. This descriptor also provides excellent results in splitting
the data.

<p align="center"><img src="svgs/409746c2b1409d7e8eca82d64144b4ca.svg" align=middle width=503.1873pt height=115.66203pt/></p>

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
<img src="svgs/2d77498996907f203ec51c0fb3344348.svg" align=middle width=42.1311pt height=24.6576pt/>. This is easily determined using Bayes rule as follows:
<p align="center"><img src="svgs/8782de4a0e653dc81e0540df101a395c.svg" align=middle width=136.32201pt height=38.834895pt/></p> If we now assume that the prior
probability <img src="svgs/a4d84573c86a53741c25b8514e89682d.svg" align=middle width=28.16979pt height=24.6576pt/> is relatively flat over all the classes, we can
simplify this to: <p align="center"><img src="svgs/d0feac63c79c98eb2be926fb091a41db.svg" align=middle width=106.179645pt height=16.438356pt/></p>

This result can now be extended to multiple features as follows:
<p align="center"><img src="svgs/488e3dabcc1410a09553225c6de4fe4f.svg" align=middle width=225.522pt height=16.438356pt/></p> This can be further simplified, as shown in figure
\[eqn:P3:indPxGx\], if we assume that each feature is independent. This
requires that features should not depend on similar data, which is not
always practically possible. However, if the dependence between features
are kept minimal, the mutual independence assumption provides a decent
approximation to the true distribution.
<p align="center"><img src="svgs/ef4c628bdebb3a012800a383a87877b5.svg" align=middle width=293.1522pt height=16.438356pt/></p> It is important to note that this algorithm
only works in one dimension at a time. Better results could be achieved
if the relations between multiple features where considered together in
a D-dimensional space. However, this would require more computation
power and more complex mechanisms. Hence there is a trade-off between
accuracy and complexity. However, the accuracy gain can often be
minimal.\
We now consider the evaluation of <img src="svgs/f7369594861336523fbfd0333e91b7f8.svg" align=middle width=42.1311pt height=24.6576pt/>. There are 2 important cases
to consider when determining this probability from the training data.
These two cases are when <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg" align=middle width=9.3951pt height=14.15535pt/> is discrete and when <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg" align=middle width=9.3951pt height=14.15535pt/> is continuous. We
first consider the case where <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg" align=middle width=9.3951pt height=14.15535pt/> is discrete and can take on one of K
values, formally denoted <img src="svgs/a2f80215322300fa4f5598a7c6118db8.svg" align=middle width=133.754775pt height=24.6576pt/>. The
<img src="svgs/f7369594861336523fbfd0333e91b7f8.svg" align=middle width=42.1311pt height=24.6576pt/> can be simply given as:
<p align="center"><img src="svgs/07d9c278bf3d9a82b6aab7496420d389.svg" align=middle width=160.981095pt height=34.177275pt/></p> Where <img src="svgs/9870b696cba5fe7a4070351492f3c796.svg" align=middle width=22.82181pt height=14.15535pt/> is the
count of the data observed having value <img src="svgs/b410e43d5465942d1e7ea4c2a547fb51.svg" align=middle width=43.93158pt height=14.15535pt/> and class <img src="svgs/0c7fe00fd6245b909fda248f712493ed.svg" align=middle width=42.249735pt height=14.15535pt/>. We
have also defined <img src="svgs/1ff67145186e685c345bfb2d86a168ff.svg" align=middle width=14.266725pt height=14.15535pt/> as the count of data observed belonging to
class <img src="svgs/5b4e948631c62d0fd9a96da246b0e5c3.svg" align=middle width=13.218315pt height=14.15535pt/>.\
We now turn our attention to the case of <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg" align=middle width=9.3951pt height=14.15535pt/> as a continuous variable.
In order to evaluate <img src="svgs/f7369594861336523fbfd0333e91b7f8.svg" align=middle width=42.1311pt height=24.6576pt/>, we need to make an assumption on the
expected form of this probability distribution. We assume that the data
takes on the form of a Gaussian distribution. This distribution appears
often due to the central limit theory. However, various features can
have a multi-modal distribution. Therefore, it is possible to miss some
detail in the distribution by making this assumption. It still produces
excellent results when put to the test. With the assumption on the form
of <img src="svgs/f7369594861336523fbfd0333e91b7f8.svg" align=middle width=42.1311pt height=24.6576pt/>, we can show that:
<p align="center"><img src="svgs/d16d5e240c239e2ed8cd172b5333c728.svg" align=middle width=158.546355pt height=18.31236pt/></p> Where we can use the
following equations to estimate <img src="svgs/ee5057e24b82e00c4e423fa0e28f58ab.svg" align=middle width=15.77961pt height=14.15535pt/> and <img src="svgs/0113a57787e1824d99cfdfd073dc0f8d.svg" align=middle width=16.535475pt height=26.76201pt/> from the
training data within each class: <p align="center"><img src="svgs/85d93c1d12d7b301c7b24563f46797f9.svg" align=middle width=190.0404pt height=105.84057pt/></p>
We can the normalise <img src="svgs/f7369594861336523fbfd0333e91b7f8.svg" align=middle width=42.1311pt height=24.6576pt/> over all the classes to calculate the
value of <img src="svgs/c745b9b57c145ec5577b82542b2df546.svg" align=middle width=10.5765pt height=14.15535pt/>. It is important to notice the minus 1 in the
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
classes across the columns and the <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg" align=middle width=9.3951pt height=14.15535pt/> across the rows. Each new
observation in the training set then just increments the counter in the
relevant row and column. One important adjustment is to set the initial
count of this matrix to 1. This is done to avoid a ’0’ probability
occurring which can greatly influence the overall probability. The
initialisation and training for discrete variable are given in
algorithms \[alg:DI\] and \[alg:DT\].

<img src="svgs/c20c67a3f74af65202fb064bbf9b1ae5.svg" align=middle width=385.086405pt height=22.83138pt/>
<img src="svgs/ee4a105cf114827a269db4340fc0264e.svg" align=middle width=373.005105pt height=22.83138pt/>
<img src="svgs/40c29252c7a328c1d585f85250296aa6.svg" align=middle width=434.474205pt height=22.83138pt/>

<img src="svgs/e3aaf728764cc78a32d014ae634ff66c.svg" align=middle width=239.698305pt height=22.83138pt/>
<img src="svgs/9a7dedc5a5b29258afa8721b9c1a155e.svg" align=middle width=201.342405pt height=22.83138pt/>
<img src="svgs/51f782cdc465d2a177c7f2f93d5053db.svg" align=middle width=382.774755pt height=24.6576pt/>

When training continuous variables, we need to consider all the data as
a whole to calculate the mean and variance. Due to this, the training is
split into two parts. The first part collects all the data associated
with each class and the second part calculated the mean and average
within each class. Algorithms \[alg:CI\], \[alg:CO\] and \[alg:CT\] show
how this could be implemented.

<img src="svgs/c20c67a3f74af65202fb064bbf9b1ae5.svg" align=middle width=385.086405pt height=22.83138pt/>
<img src="svgs/ee4a105cf114827a269db4340fc0264e.svg" align=middle width=373.005105pt height=22.83138pt/>
<img src="svgs/2afa899669e8565358ab9feeb3858e4f.svg" align=middle width=431.425005pt height=22.83138pt/>

<img src="svgs/e3aaf728764cc78a32d014ae634ff66c.svg" align=middle width=239.698305pt height=22.83138pt/>
<img src="svgs/ef3dd355f918af0715432084576214a1.svg" align=middle width=198.994455pt height=24.6576pt/>

<img src="svgs/5f2c468116b9576ea14e91699b8115d8.svg" align=middle width=402.678705pt height=22.83138pt/>
<img src="svgs/db8d2faf6a3e533b61d7ff5de26efba9.svg" align=middle width=432.550305pt height=26.76201pt/>

### Predictions

We now show how the trained classifiers can be used to make predictions.
To do this we first need to calculate the <img src="svgs/f7369594861336523fbfd0333e91b7f8.svg" align=middle width=42.1311pt height=24.6576pt/> for both discrete and
continuous variables. Algorithms \[alg:DP\] and \[alg:CP\] provide
possible pseudo code for achieving this. One we know the probability for
each input <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg" align=middle width=9.3951pt height=14.15535pt/>, we can simply multiply all of these together to get the
total probability <img src="svgs/a825e4855e3535426af74cb2355a4d9c.svg" align=middle width=89.93028pt height=24.6576pt/>. Due to equation \[eqn:cGxpxGc\], we
can now assign the point given by <img src="svgs/b0ea07dc5c00127344a1cad40467b8de.svg" align=middle width=9.97722pt height=14.61207pt/> by the class that gives
the maximum value for <img src="svgs/a825e4855e3535426af74cb2355a4d9c.svg" align=middle width=89.93028pt height=24.6576pt/>.

<img src="svgs/3f77f7e90b4f64ffe43c4a046ba30bb1.svg" align=middle width=199.121505pt height=22.83138pt/>

<img src="svgs/cfed4262790434fac4d3a46797395da8.svg" align=middle width=47.424795pt height=21.18732pt/> <img src="svgs/19391aac969426fec621ae3b3f60e7d4.svg" align=middle width=238.465755pt height=22.83138pt/>
*loop over classes c*:
<img src="svgs/e3aaf728764cc78a32d014ae634ff66c.svg" align=middle width=239.698305pt height=22.83138pt/>
<img src="svgs/bad67fb0e6b9742fd3fe3aad2143ee1a.svg" align=middle width=210.988305pt height=24.6576pt/>
<img src="svgs/2fde5887c05280c65f733eb7768815c0.svg" align=middle width=95.75346pt height=19.17828pt/>
<img src="svgs/ed78335be29a787e54e711c84643a6da.svg" align=middle width=173.662005pt height=24.6576pt/>
<img src="svgs/01b4e2180bda5d728133476fb1b86d90.svg" align=middle width=94.33776pt height=24.6576pt/>

<img src="svgs/fbdda9348c1048c45073f132e0011d54.svg" align=middle width=50.13129pt height=22.46574pt/> <img src="svgs/19391aac969426fec621ae3b3f60e7d4.svg" align=middle width=238.465755pt height=22.83138pt/>
*loop over classes c*:
<img src="svgs/e3aaf728764cc78a32d014ae634ff66c.svg" align=middle width=239.698305pt height=22.83138pt/>
<img src="svgs/a8b002b7d42dad344302f89df97b7f69.svg" align=middle width=137.19651pt height=26.76201pt/> <img src="svgs/8521ac960445404787f613640046683e.svg" align=middle width=106.450905pt height=22.46574pt/>
<img src="svgs/4caf8c54171f67f5383c7fdbf9149e09.svg" align=middle width=178.946955pt height=24.6576pt/> <img src="svgs/a6401614e9254ce613c55a750743fbbb.svg" align=middle width=97.04409pt height=24.6576pt/>

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

1.  A 10<img src="svgs/bdbf342b57819773421273d508dba586.svg" align=middle width=12.78552pt height=19.17828pt/>10 matrix is initialised to 0. This is used as the
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
and the overall accuracy is determined to be <img src="svgs/0fddbf0ea690db55def387f929f95753.svg" align=middle width=51.14175pt height=24.6576pt/>.

  -- --- ----- ------ ----- ----- ----- ----- ----- ----- ----- -----
                                                                
         0     1      2     3     4     5     6     7     8     9     
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
of <img src="svgs/0fddbf0ea690db55def387f929f95753.svg" align=middle width=51.14175pt height=24.6576pt/> and a mean certainty of <img src="svgs/a1b2964f892449bdee0c6639571cbb2a.svg" align=middle width=51.14175pt height=24.6576pt/>.

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
is equivalent to <img src="svgs/5370fb7b7c96cce03af9cf09312d11d1.svg" align=middle width=42.922605pt height=24.6576pt/>. When considering the similarity of a ’7’ and
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
have an accuracy of at least <img src="svgs/67d99e032b2a771a3cc5fc71b2f2b1b1.svg" align=middle width=30.137085pt height=24.6576pt/> for most real world data. However,
with the MNIST dataset, the classifier obtained an average accuracy
<img src="svgs/0fddbf0ea690db55def387f929f95753.svg" align=middle width=51.14175pt height=24.6576pt/>. This is poor when compared to recent algorithms, which can
achieve error rates of only <img src="svgs/d24220126879007f354add4d9e2687fa.svg" align=middle width=42.922605pt height=24.6576pt/>[@lecun]. However, these modern
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
<img src="svgs/0fddbf0ea690db55def387f929f95753.svg" align=middle width=51.14175pt height=24.6576pt/>.

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
