//
// Created by cmh on 2017. 11. 4..
//

#ifndef IMPORVEDNEURALNETWORK_NETWORK_H
#define IMPORVEDNEURALNETWORK_NETWORK_H

#include "Matrix.hpp"
#include <functional>
#include <utility>

typedef std::pair< std::vector<Matrix>, std::vector<Matrix> > Grads;

class Block;
class NeuralNetwork;

enum ActivationFuncs{
    None,
    Sigmoid,
    Step_Function,
    SoftMax,
    ReLU
};

enum ErrorFuncs{
    ACE,
    MSE
};

class ActivationLayer{
public:

    virtual Matrix feedforward(const Matrix &X) = 0;
    virtual Matrix feedbackward(const Matrix &input) = 0;
    virtual Matrix& GetLast_y() = 0;
};

class AffineLayer{
public:
    explicit AffineLayer(size_t output_size);

    Matrix feedforward(const Matrix &X);
    Matrix feedbackward(const Matrix &input);

    void SetUpSize(size_t Input_size);
    void SetUpSizeWithStd(size_t Input_size, int node_num);

    Matrix& GetW();
    Matrix& GetB();
    Matrix& GetDw();
    Matrix& GetDb();
    Matrix& GetLast_x();
    Matrix& GetLast_y();
    size_t GetOutputSize();
    int GetNumOfNodes();

    //실험
    int PlusStack = 0;
    int MultiplyStack = 0;
private:
    Matrix W;
    Matrix b;
    size_t Output_Size;

    //역전파법
    Matrix Dw;
    Matrix Db;
    Matrix Last_x;
    Matrix Last_y;


};

class SigmoidLayer : public ActivationLayer{
public:
    Matrix feedforward(const Matrix &X) override;
    Matrix feedbackward(const Matrix &input) override;
    Matrix& GetLast_y() override ;

private:
    Matrix Last_y;
};

class SoftMaxLayer : public ActivationLayer{

};

class Block{
public:
    Block(AffineLayer layer, ActivationFuncs Acts) : Affine(std::move(layer))
    {
        switch (Acts)
        {
            case Sigmoid:
                ActL =new SigmoidLayer();
                break;
            default:
                break;
        }
    }


    Matrix feedforward(const Matrix &X);
    Matrix feedbackward(const Matrix &output);
    Matrix feedbackward(double input);

    AffineLayer& GetAffine();
    ActivationLayer& GetActL();
private:
    AffineLayer Affine;
    ActivationLayer* ActL;
};

class NeuralNetwork{
public:
    explicit NeuralNetwork(size_t Input_Size);

    NeuralNetwork &operator<<(Block &layer);

    Matrix predict(Matrix X);

    double loss(Matrix &x, Matrix &t);
    Grads gradient(Matrix x, Matrix t);
    Grads numerical_gradient(Matrix x, Matrix t);

    void Train(std::pair<Matrix, Matrix> Train_Data, double Rate);
    void Trains(std::vector<std::pair<Matrix, Matrix>> Train_Datas, double Rate);
    void Diff(std::pair<Matrix, Matrix> Train_Data);

    double Last_loss = 0; // for debug

    int GetPlusStacks() const ;
    int GetMultiplyStacks()const ;
    Matrix& GetLast_t();
private:

    class lossLayer{
    public:
        double feedforward(const Matrix &x, const Matrix &t);
        Matrix feedbackward(const Matrix &Last_y, const Matrix &Last_t);

    };

    Matrix Last_t;
    size_t InputSize;
    void OptimizeSizes();
    std::vector<Block> Blocks;//블럭들의 리스트
    lossLayer LossLayer;//손실레이어

};
#endif //IMPORVEDNEURALNETWORK_NETWORK_H
