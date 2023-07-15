# Modeling-PUFs

## Abstract
Physical unclonable functions (PUFs) are hardware devices that produce unique responses to challenges to authenticate users. However, it has been demonstrated that these devices are vulnerable to machine learning modeling attacks in which correct responses to challenges can be predicted with high accuracy. This development project recreates previously successful modeling techniques and assesses their capability on up to 5-XORed arbiter PUFs by measuring each model’s accuracy and training time. It demonstrates that deep learning models are the most effective method and can learn each APUF composition whereas simpler models like logistic regression and support vector machines are successfully defended by XORing APUFs.

Please see the associated [research paper](./Modeling_Arbiter_PUFs.pdf) to learn more about PUFs, the methods used, and the results.

## References
* Riedmiller, M., & Braun, H. (1993). A direct adaptive method for faster backpropagation learning: The RPROP algorithm. *IEEE International Conference on Neural Networks*. https://doi.org/10.1109/icnn.1993.298623
* Rührmair, U., Sehnke, F., Sölter, J., Dror, G., Devadas, S., & Schmidhuber, J. (2010, October). Modeling attacks on physical unclonable functions. In *Proceedings of the 17th ACM conference on Computer and communications security* (pp. 237-249).
* Santikellur, P., Bhattacharyay, A., & Chakraborty, R. S. (2019). Deep learning based model building attacks on arbiter PUF compositions. *Cryptology ePrint Archive*.

In models.py, RPropLogisticRegression contains code from [Logistic-Regression-From-Scratch](https://github.com/casperbh96/Logistic-Regression-From-Scratch).

MIT License

Copyright (c) 2021 Casper Bøgeskov Hansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

It has been modified by me, Jakub Vogel.