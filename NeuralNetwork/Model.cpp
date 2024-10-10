#include "Model.h"
const int maxThread = 30;

Model& Model::layer(Layer& layer)
{
    this->BiasMomentum.push_back(Eigen::VectorXd::Zero(layer.Bias->size()));
    this->WeightMomentum.push_back(Eigen::MatrixXd::Zero(layer.outputCount, layer.inputCount));
    this->layers.push_back(layer);
    return (*this);
}


Eigen::VectorXd Model::getOutput(Eigen::VectorXd& input)
{
    Eigen::VectorXd output = input;
    for (auto& i : this->layers)
    {
        output  = i.activation->activationN(i.getOutPut(output));
    }
    return output;
}

void startTrainginThread(double* totalCost, Model* model, const std::pair<Eigen::VectorXd, Eigen::VectorXd>*entry,int data_size,std::vector<Eigen::VectorXd>* BiasGradient, std::vector<Eigen::MatrixXd>* WeightGradient)
{
    *totalCost += model->backpropagation(entry->first, entry->second,data_size,*BiasGradient,*WeightGradient);
}

double Model::train(Dataset& data,int batchSize)
{    
    const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& datas = data.getAll();
    double totalCost = 0;
    int trainingCount = datas.size();
    std::vector<std::thread> threads;
    std::vector<Eigen::VectorXd> BiasGradient(layers.size());
    std::vector<Eigen::MatrixXd> WeightsGradient(layers.size());

    for (int i = 0; i < layers.size(); i++) 
    {
        BiasGradient[i] = Eigen::VectorXd::Zero(layers[i].outputCount);
        WeightsGradient[i] = Eigen::MatrixXd::Zero(layers[i].outputCount, layers[i].inputCount);

    }
    int counter = 0; 
    int batchCounter = 0;
    int batchCount = datas.size() / batchSize + 1;
    

    for(int k = 0; k< datas.size(); k++)
    startTrainginThread(&totalCost, this, &datas[k], datas.size(), &BiasGradient, &WeightsGradient);

    for (auto& i : threads) i.join();

    for (int i = 0; i < BiasGradient.size(); i++) 
    {
        *this->layers[i].weights = *this->layers[i].weights -  WeightsGradient[i] * learningRate + this->WeightMomentum[i];
        *this->layers[i].Bias = *this->layers[i].Bias-  BiasGradient[i] * learningRate + this->BiasMomentum[i];

        this->WeightMomentum[i] = WeightsGradient[i] * -this->learningRate + this->WeightMomentum[i] * this->momentumMultiplier;
        this->BiasMomentum[i] = BiasGradient[i] * -this->learningRate + this->BiasMomentum[i] * this->momentumMultiplier;
    }
    double averageCost = totalCost / datas.size();
    //std::cout <<"cost: " << averageCost << "\n";
    return averageCost;
}

double Model::trainInBatch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& input, int start, int stop,std::vector<Eigen::VectorXd>& Bias, std::vector<Eigen::MatrixXd>& Weights)
{
    //std::cout << "Started Batch " << start << " " << stop<< "\n";
    double totalCost = 0;
    std::vector<std::thread> threads;

    for (int i = 0; i < layers.size(); i++)
    {
        Bias[i] = Eigen::VectorXd::Zero(layers[i].outputCount);
        Weights[i] = Eigen::MatrixXd::Zero(layers[i].outputCount, layers[i].inputCount);
    }
    for (int i = 0; i < layers.size(); i++)
    {
        Bias[i] = Eigen::VectorXd::Zero(layers[i].outputCount);
        Weights[i] = Eigen::MatrixXd::Zero(layers[i].outputCount, layers[i].inputCount);

    }
    int counter = 0;
    int batchCounter = 0;

    for(int i = start; i< stop; i++)
        startTrainginThread(&totalCost, this, &input[i], input.size(), &Bias, &Weights);

    for (auto& i : threads) i.join();
    double averageCost = totalCost /(stop - start);
    //std::cout <<"cost: " << averageCost << "\n";
    return averageCost;
}

double Model::batchingTraining(Dataset& data, int batchSize)
{
    const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& datas = data.getAll();
    std::vector<Eigen::VectorXd> BiasGradient(layers.size());
    std::vector<Eigen::MatrixXd> WeightsGradient(layers.size());
    int current_start = 0, current_stop = batchSize > datas.size() ? datas.size() : batchSize;
    int batchCount = 0;
    double totalCost = 0;
    while (current_start < datas.size()) 
    {
        totalCost+= trainInBatch(datas, current_start, current_stop> datas.size()?datas.size(): current_stop, BiasGradient, WeightsGradient);
        current_start += batchSize;
        current_stop += batchSize;
        batchCount++;

        for (int i = 0; i < BiasGradient.size(); i++)
        {
            *this->layers[i].weights = *this->layers[i].weights - WeightsGradient[i] * learningRate + this->WeightMomentum[i];
            *this->layers[i].Bias = *this->layers[i].Bias - BiasGradient[i] * learningRate + this->BiasMomentum[i];

            this->WeightMomentum[i] = WeightsGradient[i] * -this->learningRate + this->WeightMomentum[i] * this->momentumMultiplier;
            this->BiasMomentum[i] = BiasGradient[i] * -this->learningRate + this->BiasMomentum[i] * this->momentumMultiplier;
        }
    }
    return totalCost / batchCount;
}

void Model::BackUp(std::filesystem::path path_to_file)
{
    //Binary Plz
}

void Model::LoadBackUp(std::filesystem::path path_to_file)
{

}



double Model::backpropagation(const Eigen::VectorXd& input,const Eigen::VectorXd& expected_output,int data_size, std::vector<Eigen::VectorXd>& BiasGradient, std::vector<Eigen::MatrixXd>& WeightGradient)
{
    std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>> outputs;
    for (auto& i : this->layers) 
    {
        Eigen::VectorXd temp;
        if (outputs.empty())
            temp = i.getOutPut(input);
        else
            temp = i.getOutPut(outputs.back().first);
        outputs.push_back({ i.activation->activation(temp),i.activation->derivative(temp)});
    };
    Eigen::VectorXd e = (outputs.back().first - expected_output).array() *  outputs.back().second.array() * 2;
    for (int i = outputs.size()-1; i>=0 ; i--) 
    {
        Eigen::VectorXd newE = ((*this->layers[i].weights).transpose() * e).array() ;
        Eigen::MatrixXd JoverW;
        if (i != 0)JoverW = outputs[i - 1].first * e.transpose();
        else JoverW = input * e.transpose();
        //std::cout << "LayerWeight\n" << e << "\n\n\LayerBias\n" << *this->layers[i].Bias << "\n\n";
        //std::cout << "Weight\n" << weight << "\n\nnew E\n" << newE << "\n\nJoverE\n" << JoverW << "\n\n";
        WeightGradient[i] += JoverW.transpose();
        BiasGradient[i] +=  e;
        e = newE;
    }


    double cost = 0;
    for (int i = 0; i < outputs.back().first.size(); i++)
    {
        double temp = outputs.back().first(i) - expected_output(i);
        cost += temp * temp;
    }
    return cost;
}
