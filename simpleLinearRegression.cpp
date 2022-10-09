// contains implementation for simple linear regression algorithm

#include <iostream>
#include <math.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <list>
using namespace std;


class simpleLinearRegression{
private:
    int num_epochs;
    double alpha;
    xt::xarray<double> X;
    xt::xarray<double> y;
    xt::xarray<double> W;
    xt::xarray<double> b;
    xt::xarray<double> X0;
    int datapoints = 0;
    list<double> losses;
    list<double> pred_list;

public:

    simpleLinearRegression(xt::xarray<double> input_feature, xt::xarray<double> output, const int epochs, float learningRate,int dataSize) {
        num_epochs = epochs;
        alpha = learningRate;
        int datapoints = dataSize;
        X0 = xt::ones<double>({ datapoints,1 });
        X = input_feature;
        y = output;
        W = xt::random::randn<double>({ 1, 1 });
        b = xt::random::randn<double>({ 1, 1 });
        

    }

    void updateWeights(double loss) {
        for (int datapoint = 0; datapoint < datapoints; datapoint++) {
            W(0, 0) -= alpha * loss * X(datapoint);
        }
    }

    double cal_loss() {
        // calculate mean squared loss

        // y_hat = W*X + b*X0
        double epochLoss = 0.0;
        xt::xarray<int>indices = xt::arange(0, datapoints);
        xt::random::shuffle(indices);
        for (int datapoint : indices) {
            double y_hat = W(0,0) * X(datapoint) + b(0,0) * X0(datapoint);
            double loss = pow((y_hat-y(datapoint)), 2)/2;
            updateWeights(loss);
            epochLoss+=loss; 
        }
        losses.push_back(epochLoss); // store loss

        return epochLoss;
    }

    void train() {

        double loss = 0.0;
        for (int epoch = 1; epoch < num_epochs; epoch++) {
            loss = cal_loss();
            cout << "Epoch: " << epoch << "Loss: " << loss << endl;
        }

    }

    void predict(xt::xarray<double> X_test, int testSize) {
        for (int i = 0; i <= testSize; i++) {
            double pred = W(0, 0) * X_test(i) + b(0, 0);
            pred_list.push_back(pred);
            cout << "Point: " << i << "Pred: " << pred << endl;
        }

    }


    void getParams() {
        cout << "W: " << W(0, 0) << endl;
        cout << "b: " << b(0, 0) << endl;
    }

    void getPredLoss() {
        double total_loss = 0.0;
        for (double lossValue: pred_list) {
            cout << lossValue << endl;
            total_loss += lossValue;
        }
        cout << total_loss << endl;
    }

};

int main()
{

    // create simple train data
    int dataSize = 50;
    xt::xarray<double> X = xt::arange(1, dataSize + 1); // independent feature
    X.reshape({ dataSize ,1 });
    xt::xarray<double> y = 2 * X + 3; // dependent feature
    const int epochs = 100;
    const double learningRate = 0.001;

    // initialize regressor
    simpleLinearRegression regressor(X,y,epochs,learningRate,dataSize);

    // train the model
    regressor.train();

    //// create test data
    //int testSize = 100;
    //xt::xarray<double> X_test = xt::arange(dataSize + 1, testSize); // independent feature
    //X.reshape({ testSize - dataSize - 1 ,1 });
    //xt::xarray<double> y_test = 2 * X_test + 3; // dependent feature
    

    
    return 0;
}