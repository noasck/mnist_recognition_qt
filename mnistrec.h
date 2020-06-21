#ifndef MLP_FOR_MNIST_MNISTREC_H
#define MLP_FOR_MNIST_MNISTREC_H
#include <Eigen/Dense>
#include <vector>

namespace MLP
{
    struct Result
    {
        int predict;
        float pred_pers;
        float error;
        std::vector<int> image;
    };
    struct Result test(int sample);
    void train(int sample);
}

#endif //MLP_FOR_MNIST_MNISTREC_H
