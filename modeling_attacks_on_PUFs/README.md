# Testing *Modeling attacks on physical unclonable functions* on New Data

## Description
This sub-section focuses on applying the methods of the seminal 2010 *Modeling attacks on physical unclonable functions* that achieved 99% accuracy on up to 5-XOR APUFs. Some edits were made to the original code provided in the paper to allow it to run with present day libraries. On their generated 2-XOR APUF, I was able to recreate the high accuracy (98.88%) they reported.

However, I wanted to try their methodology on the data I was working with now. With lots of code analysis and effort, I was able to get their tools to run on that data. But, the accuracy was not what I expected. The LR model with RProp gradient descent only achieved 52.64% accuracy on the 2-XOR APUF from my data.

## Source Work
Rührmair, U., Sehnke, F., Sölter, J., Dror, G., Devadas, S., & Schmidhuber, J. (2010, October). Modeling attacks on physical unclonable functions. In *Proceedings of the 17th ACM conference on Computer and communications security* (pp. 237-249).