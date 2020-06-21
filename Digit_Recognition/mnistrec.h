#ifndef MLP_FOR_MNIST_MNISTREC_H
#define MLP_FOR_MNIST_MNISTREC_H
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "mnistrec.h"
#include <QApplication>

using namespace std;

namespace MLP
{


#include <ctime>
#include <chrono>
using namespace std::chrono;
using namespace std;

typedef float (* node_function_t)(float a) ;

typedef float (* loss_function_t)(float y, float y_hat);

class Functions
{
private:
    static float loss_function(float y, float y_hat)
    {
        return (y - y_hat)*(y - y_hat);
    }
    static float der_loss_function(float y, float y_hat)
    {
        return 2*(y-y_hat);
    }
    static float node_function(float x)
    {
        return tanh(0.01*x); //hyperbolic tangence
        // return
    }
    static float der_node_function(float x)
    {
        float der = 1/(cosh(0.01*x)*cosh(0.01*x));

        return der;
    }

public:
    node_function_t f = node_function;
    node_function_t df = der_node_function;
    loss_function_t l = loss_function;
    loss_function_t dl = der_loss_function;
};

class Layer {
public:
    Eigen::VectorXf result;
    Eigen::MatrixXf w;
    Eigen::VectorXf dres;
    Layer (int layer_nodes_count, int prev_nodes_count) {
        w = Eigen::MatrixXf::Random(layer_nodes_count, prev_nodes_count);

    }

    float operator[](int index){
        return this->result[index];
    }

    Eigen::VectorXf computer_out(const Eigen::VectorXf& x, node_function_t f, node_function_t df) {
        Eigen::VectorXf res = w * x;
        // if(res.size()==784){
        //   std::cout << w*x << '\n';
        // }
        dres.resize(res.size());
        for (int i = 0; i < w.rows(); i++) {
            dres[i] = df(res[i]);
            res[i] = f(res[i]);
        }
        result = res;
        return res;
    }


};


class Network
{
protected:
    Layer input;
    Layer hl ;
    Layer output;
    float learning_rate = 0.01;
    // float learning_rate = 0.002;
    float deviation = 0.02;
    // float deviation = 0.02;
    Functions functions;

public:
    Network(int input_count, int hidden_layer, int output_count, Functions functools)
    : input(input_count, input_count), hl(hidden_layer, input_count),
    output(output_count, hidden_layer){
        functions = functools;
    }


    Eigen::VectorXf learn_sample(const Eigen::VectorXf& x, const Eigen::VectorXf& out)
    {
        gradient_descent(x, out);
        return compute_result(x);
    }

    Eigen::VectorXf compute_input(const Eigen::VectorXf& x)
    {
        return input.computer_out(x, functions.f, functions.df);
    }

    Eigen::VectorXf compute_hidden(const Eigen::VectorXf& x)
    {
        return hl.computer_out(x, functions.f, functions.df);
    }

    Eigen::VectorXf compute_output(const Eigen::VectorXf& x)
    {
        return output.computer_out(x, functions.f, functions.df);
    }

    Eigen::VectorXf compute_result(const Eigen::VectorXf& x)
    {
      // std::cout << x << '\n';
        return compute_output(compute_hidden(compute_input(x)));
    }


    void set_learning_rate(float rate)
    {
        learning_rate = rate;
    }

    void set_deviation(float max_deviation)
    {
        deviation = max_deviation;
    }

    void gradient_descent(const Eigen::VectorXf& x, const Eigen::VectorXf& out, int max_count = 4)
    {
        int count = 0;
        float D, D1;
        Eigen::VectorXf D_v, dDdz;
        Eigen::MatrixXf dw, du, dv;
        for (size_t i = 0; i < out.size(); i++) {
          if(out[i]!=0)
          {
            std::cout << " Out: "<< i << '\n';
            std::cout << compute_result(x)[i] << '\n';
          }
        }
        // compute_result(x);
        do {
            D_v = compute_loss_function(out);
            dDdz = compute_der_d_z(out);
            D = D_v.sum();
            // std::cout << grad_output(dDdz) << '\n';
            descent(grad_output(dDdz), grad_hidden_2(dDdz), grad_input_2(dDdz, x));
            compute_result(x);
            D1 = compute_loss_function(out).sum();
            std::cout << "D =" << D <<' '<<D1 << "Difference: " <<D-D1<<'\n';
            count++;
        }
        while (
          // D>1
        // );
          D - D1 > deviation &&
           count<max_count);
    }

    void descent(const Eigen::MatrixXf& dw, const Eigen::MatrixXf& du, const Eigen::MatrixXf& dv )
    {
        input.w -= dv;
        hl.w -= du;
        output.w -= dw;
    }

    Eigen::VectorXf compute_loss_function(const Eigen::VectorXf& out)
    {
        Eigen::VectorXf D(output.result.size());
        for (int i = 0; i < output.result.size(); i++)
        {
            D[i] = functions.l(output[i], out[i]);
        }
        return D;
    }

    Eigen::VectorXf compute_der_d_z(const Eigen::VectorXf& out)
    {
        Eigen::VectorXf res(output.result.size());

        for (int i = 0; i < output.result.size(); ++i) {
            res[i] = functions.dl(output.result[i], out[i]);
        }
        return res;
    }


    Eigen::MatrixXf grad_output_2(const Eigen::VectorXf& dDdZ)
    {
        auto start = high_resolution_clock::now();
        Eigen::MatrixXf dw(output.w.rows(), output.w.cols());

        for (int i = 0; i < output.w.rows(); i++)
        {
            for(int j = 0; j < output.w.cols(); j++)
            {
                dw(i,j) = dDdZ[i]*hl.result[j]*output.dres[i];
            }
        }
        // std::cout << "grad_out finished in " <<(high_resolution_clock::now() - start).count()<< '\n';
        return dw*learning_rate;
    }

    Eigen::MatrixXf grad_output(const Eigen::VectorXf& dDdZ)
    {
        auto start = high_resolution_clock::now();
        Eigen::MatrixXf dw = dDdZ * hl.result.transpose();
        // std::cout << dw << '\n';
        dw  = output.dres.asDiagonal() *dw;
        // std::cout << "grad_out finished in " <<(high_resolution_clock::now() - start).count()<< '\n';
        return dw*learning_rate;
    }

    Eigen::MatrixXf grad_hidden(const Eigen::VectorXf& dDdZ)
    {
      auto start = high_resolution_clock::now();

        Eigen::MatrixXf du(hl.w.rows(), hl.w.cols());
        du.setZero();
        for (int i = 0; i < hl.w.rows(); i++)
        {
            for(int j = 0; j < hl.w.cols(); j++)
            {
                for (int k = 0; k < output.w.rows(); k++)
                {
                  // std::cout << hl.w.rows()<<' '<<j<<' '<<k << '\n';
                    du(i,j) += dDdZ[k]*output.w(k,i)*output.dres[k];
                }
                du(i,j) *= input.result[j]*hl.dres[i];
            }
        }
        // std::cout << "grad_out finished in " <<(high_resolution_clock::now() - start).count()<< '\n';

        return du*learning_rate;
    }
    Eigen::MatrixXf grad_hidden_2(const Eigen::VectorXf& dDdZ)
    {
      auto start = high_resolution_clock::now();

        Eigen::MatrixXf du(hl.w.rows(), hl.w.cols());
        du =hl.dres.asDiagonal()* (((output.dres.asDiagonal()*output.w).transpose() * dDdZ)*input.result.transpose());

        // std::cout << "grad_out finished in " <<(high_resolution_clock::now() - start).count()<< '\n';
        // std::cout << du << '\n';
        return du*learning_rate;
    }

    Eigen::MatrixXf grad_hidden_3(const Eigen::VectorXf& dDdZ)
    {
      auto start = high_resolution_clock::now();

        Eigen::MatrixXf du(hl.w.rows(), hl.w.cols());
        du.setZero();
        Eigen::MatrixXf temp = output.dres.asDiagonal()*output.w;
        temp = dDdZ.asDiagonal()*temp;
        float temp_sum;
        for (int i = 0; i < hl.w.rows(); i++)
        {
            temp_sum = temp.col(i).sum();
            for(int j = 0; j < hl.w.cols(); j++)
            {
                du(i,j) =temp_sum* input.result[j]*hl.dres[i];
            }
        }
        // std::cout << "grad_hidden finished in " <<(high_resolution_clock::now() - start).count()<< '\n';

        return du*learning_rate;
    }

    Eigen::MatrixXf grad_input(const Eigen::VectorXf& dDdZ, const Eigen::VectorXf& in)
    {

        int iwc = input.w.cols();
        Eigen::MatrixXf dv(input.w.rows(), iwc);
        dv.setZero();
        int ors = output.result.size();
        for(int i = 0; i < input.w.rows(); i++)
        {
            for(int j = 0; j < iwc; j++)
            {
                for(int k= 0; k < ors; k++)
                {
                    for(int t = 0; t < hl.w.rows(); t++)
                    {
                        dv(i,j) +=hl.w(t,i)*output.w(k,t)*hl.dres[t];
                    }
                    dv(i,j)*=dDdZ[k]*output.dres[k];
                     // std::cout << dv(i,j) << '\n';

                }
                dv(i,j)*=input.dres[i]*in[j];
            }

        }

        return dv*learning_rate;
    }
    Eigen::MatrixXf grad_input_2(const Eigen::VectorXf& dDdZ, const Eigen::VectorXf& in)
    {
      auto start = high_resolution_clock::now();
        Eigen::MatrixXf dv(input.w.rows(), input.w.cols());
        dv = input.dres.asDiagonal()*(((output.dres.asDiagonal()*(output.w*(hl.dres.asDiagonal()*hl.w))).transpose() * dDdZ)*in.transpose());

        // std::cout << "grad_in finished in " <<(high_resolution_clock::now() - start).count()<< '\n';
        // std::cout << dv << '\n';
        return dv*learning_rate;
    }


        Eigen::MatrixXf grad_input_3(const Eigen::VectorXf& dDdZ, const Eigen::VectorXf& in)
        {
          auto start = high_resolution_clock::now();

            int iwc = input.w.cols();
            Eigen::MatrixXf dv(input.w.rows(), iwc);
            // std::cout << hl.dres << '\n';
            Eigen::MatrixXf temp = hl.dres.asDiagonal()*hl.w ;
            temp =output.w*temp;
            //std::cout << temp << '\n';
            dv.setZero();
            float temp_sum;
            int ors = output.result.size();
            for(int i = 0; i < input.w.rows(); i++)
            {
              temp_sum = temp.col(i).sum();
                for(int j = 0; j < iwc; j++)
                {
                    for(int k= 0; k < ors; k++)
                    {
                        dv(i,j) *=dDdZ[k]*output.dres[k]*temp_sum ;
                    }
                    dv(i,j)*=input.dres[i]*in[j];
                }
                //std::cout << i << dv(i,0)<< '\n';
            }
            // std::cout << "grad_in finished in " <<(high_resolution_clock::now() - start).count()<< '\n';

            return dv*learning_rate;
        }


    void debug_output()
    {
      //std::cout << input.w << '\n';
    //  std::cout << hl.w << '\n';
      // std::cout << output.w << '\n';
    }

    void write_in_file()
    {
      ofstream off("weights.in");
      for (size_t i = 0; i < output.w.rows(); i++) {
        for (size_t j = 0; j < output.w.cols(); j++) {
          off<< output.w(i,j)<<endl;
        }
      }
      for (size_t i = 0; i < hl.w.rows(); i++) {
        for (size_t j = 0; j < hl.w.cols(); j++) {
          off<< hl.w(i,j)<<endl;
        }
      }
      for (size_t i = 0; i < input.w.rows(); i++) {
        for (size_t j = 0; j < input.w.cols(); j++) {
          off<< input.w(i,j)<<endl;
        }
      }
      off.close();
    }

    void read_from_file() {
      ifstream off("weights.in");
      if (off.is_open())
      {
      for (size_t i = 0; i < output.w.rows(); i++) {
        for (size_t j = 0; j < output.w.cols(); j++) {
          off>> output.w(i,j);
        }
      }
      for (size_t i = 0; i < hl.w.rows(); i++) {
        for (size_t j = 0; j < hl.w.cols(); j++) {
          off>> hl.w(i,j);
        }
      }
      for (size_t i = 0; i < input.w.rows(); i++) {
        for (size_t j = 0; j < input.w.cols(); j++) {
          off>> input.w(i,j);
        }
      }
    }
      off.close();
    }

};



  Eigen::MatrixXf read_mnist(int samples)
  {
    Eigen::MatrixXf x(28*28, samples);

      ifstream file ("t10k-images-idx3-ubyte");
      int magic_number;
      file.read((char*)&magic_number,sizeof(int));
      file.read((char*)&magic_number,sizeof(int));
      file.read((char*)&magic_number,sizeof(int));
      file.read((char*)&magic_number,sizeof(int));
      if (file.is_open())
      {
          for(int i=0;i<samples;i++)
          {
              for(int r=0;r<784;r++)
              {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                x(r, i)=(float)temp/255;
                 // std::cout <<r<<" "<< (float)temp/255 << ' '<<x(r, i)<< '\n';
              }
          }
      }
      return x;
  }

  Eigen::VectorXf read_mnist_labels(int samples)
  {
      Eigen::VectorXf out(samples);
      ifstream file ("t10k-labels-idx1-ubyte");
      int magic_number;
      file.read((char*)&magic_number,sizeof(int));
      file.read((char*)&magic_number,sizeof(int));
      if (file.is_open())
      {
          for(int i=0;i<samples;i++)
          {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            out[i] = (int)temp;

          }
      }
      return out;
  }

  struct Result
  {
    int predict;
    float pred_pers;
    float error;
    std::vector<float> image;
  };

struct Result test(int sample)
{
  Eigen::MatrixXf temp = read_mnist(sample);
  Eigen::VectorXf labels = read_mnist_labels(sample);
  Network nt(784, 256, 10, Functions());
  nt.read_from_file();
  Result result;
  Eigen::VectorXf res;
  float error = 0;
    // Eigen::VectorXf out_v(10);
    // out_v.setZero();
    // out_v[labels[sample-1]] = 1;

    Eigen::VectorXf in = temp.col(sample-1);
    // std::cout << in << '\n';
    for (size_t i = 0; i < 784; i++) {
      result.image.push_back(in[i]);
    }
    res = nt.compute_result(in);
    int max_res = 0;
    for (size_t j = 0; j < res.size(); j++) {
      if(res[j]>res[max_res]) max_res = j;
        error +=res[j];
    }
    std::cout << max_res <<" "<< labels[sample-1] <<" " << '\n';
    result.predict = max_res;
    result.pred_pers = res[max_res];
    result.error = error - res[max_res];

  return result;
}
struct Result test_custom(const Eigen::VectorXf& in)
{
  
  Network nt(784, 256, 10, Functions());
  nt.read_from_file();
  Result result;
  Eigen::VectorXf res;
  float error = 0;
    // Eigen::VectorXf out_v(10);
    // out_v.setZero();
    // out_v[labels[sample-1]] = 1;


    // std::cout << in << '\n';
    for (size_t i = 0; i < 784; i++) {
      result.image.push_back(in[i]);
    }
    res = nt.compute_result(in);
    int max_res = 0;
    for (size_t j = 0; j < res.size(); j++) {
      if(res[j]>res[max_res]) max_res = j;
        error +=res[j];
    }
    
    result.predict = max_res;
    result.pred_pers = res[max_res];
    result.error = error - res[max_res];

  return result;
}

void train(int sample)
{
  Eigen::MatrixXf temp = read_mnist(sample);
  Eigen::VectorXf labels = read_mnist_labels(sample);
  Network nt(784, 256, 10, Functions());

   for (size_t i = 0; i < sample; i++)
   {
    Eigen::VectorXf out_v(10);
    out_v.setZero();
    out_v[labels[i]] = 1;
    Eigen::VectorXf in = temp.col(i);
    nt.learn_sample(in, out_v);
    // std::cout << i+1 << '\n';
   }
   nt.write_in_file();
}

}

#endif //MLP_FOR_MNIST_MNISTREC_H
