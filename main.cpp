#include <iostream>
#include "Libraries/Network.h"

using namespace std;


int main() {



    Block b1(AffineLayer(5), Sigmoid);
    Block b2(AffineLayer(1), Sigmoid);

    NeuralNetwork Network(2);
    Network<<b1<<b2;


    int last = 0;
    int num = 0;


    cout<<"Train횟수 : ";
    cin>>num;
    for (int i = 0; i < num; ++i) {

        if(last != (int)(((double)i / num)*100)) //마지막 진행%가 현재 진행%랑 다를시(정수로 변환시킴)
        {

            cout<<(int)(((double)i / num)*100)<<"%"<<"   loss : "<<Network.Last_loss<<endl; //현재 진행 상황과 손실값이 얼마인지 출력(손실값 : 현재 출력되는 값이 정답과 얼마나 다른지 알려주는 값)
            last = (int)(((double)i / num)*100); //last갱신
        }


        Network.Trains({
                              {Matrix({0,0}),Matrix({0})},
                              {Matrix({1,0}),Matrix({1})},
                              {Matrix({0,1}),Matrix({1})},
                              {Matrix({1,1}),Matrix({0})}
                      }, 0.1);

    }

    Matrix x({0,0});
    Network.predict(x);


    cout<<"오차역전파법 사용합니다."<<endl<<endl;
    cout<<"Plus Count : " << Network.GetPlusStacks()<<endl;
    cout<<"Multiply Count : " << Network.GetMultiplyStacks()<<endl;

    /*
    Matrix x({0,0});
    cout<<"0 XOR 0 : ";
    Network.predict(x).print();
    cout<<Network.predict(x).PreviousPlusStack<<endl;
    cout<<"1 XOR 0 : ";
    x = {1,0};
    Network.predict(x).print();
    x = {0,1};
    cout<<"0 XOR 1 : ";
    Network.predict(x).print();
    x = {1,1};
    cout<<"1 XOR 1 : ";
    Network.predict(x).print();
*/



    return 0;
}
