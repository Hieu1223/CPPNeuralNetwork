#include <iostream>
#include<time.h>
#include"Eigen/Dense"
#include"Layer.h"
#include"Model.h"

class Num : public Dataset 
{
public:
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> datas;

    Num() 
    {           
        for (int i = 0; i < 50; i++) 
        {
            Eigen::VectorXd input(21);
            input.setZero();
            Eigen::VectorXd output(1);
            output(0) = i%3 == 0;
            input(i%10) = 1;
            input(10 + i / 10) = 1;
            input(20) = i % 10 + i / 10;
            datas.push_back({input, output});
        }
    }

public:
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& getAll()
    {
        return this->datas;
    }
};


int main()
{
    time_t m_time = time(nullptr);
    tm t;
    srand(localtime_s(&t, &m_time));
    Activation*test = new Activation();
    Layer testLayer(21,5,test);
    Layer testLayer1(5, 5,test);
    Layer testLayer2(5, 2,test);
    Layer testLayer3(2, 1, test);
    Model model = Model().layer(testLayer).layer(testLayer1).layer(testLayer2).layer(testLayer3);

    
    Num testDataset;
    float cost = 1;
    std::cout << "Training\n";
    while (cost > 0.1) 
    {
        cost = model.batchingTraining(testDataset,20);
        std::cout<< "cost: " << cost << "\n";
    };


    std::cout << "Testing" << "\n";
    int testCount = 100;
    for (int i = 0; i < testCount; i++) 
    {
        Eigen::VectorXd input(21);
        input.setZero();
        int random = rand() % 100;
        input(random%10) = 1;
        input(10 + random / 10) = 1;
        input(20) = random % 10 + random / 10;
        Eigen::VectorXd output =  model.getOutput(input);
        int result = output(0) > 0.5f ? 1 : 0;
        std::cout << random << " " << result << "\n";
    }

}
