#include <torch/torch.h>
#include <iostream>
#include <chrono>

using std::chrono::high_resolution_clock;

auto t1 = high_resolution_clock::now();

// Batch Size
const int64_t N = 64;

// Input Layer
const int64_t D_in = 1000;

// Hidden Layer
const int64_t H = 100;

// Output Layer
const int64_t D_out = 10;

// Neural Network
struct Lab : torch::nn:module {
    // Constructor
    Lab() : linear1(D_in, H), linear2(H, D_out) {
        register_module("linear1", linear1);
        register_module("linear2", linear2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(linear1->forward(x));
        x = linear2->forward(x);
        return x;
    }

    // Variables
    torch::nn::Linear linear1;
    torch::nn::Linear linear2;
};

int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // Seed
    torch::manual_seed(1);

    // Create random values to x and y
    torch::Tensor x = torch::rand({N, D_in});
    torch::Tensor y = torch::rand({N, D_out});

    torch::Device device(torch::kCPU); //kCPU or kCUDA
    
    // Create model
    Lab model;

    model.to(device);

    float_t learning_rate = 1e-4;

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate));

    // Traning
    for (size_t epoch = 1; epoch <= 500; epoch++;) {
        auto y_pred = model.forward(x);

        auto loss = torch::mse_loss(y_pred, y.detach());

        if (epoch % 100 == 99) {
            std::cout << "Epoch: " << epoch << ". Erro" << loss << endl;
        }

        // Reset gradients
        optimizer.zero_grad();

        // Backpropagation
        loss.backward();

        // Update weights
        optimizer.step();
    }

    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Run Time = " << ms_double.count() << " ms" << endl;

    return 0;
}

