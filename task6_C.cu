#include <cmath>
#include <string>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>



// функция активации
__global__ void nn_Sigmoid(float* arr, int size)
{
	int id = threadIdx.x;

	arr[id] = 1 / (1 + exp(-arr[id]));
}



class NN
{
private:
	cublasHandle_t handle;
	float alpha, beta;
	float* weights, * biases, * output;
	int input_size, output_size;
	bool activation_true;

	// считывание весов из файла
	void read_weights(std::string pathToWeights) {
		float* host_array = new float[input_size * output_size];
		float* host_array_row = new float[input_size * output_size];

		std::ifstream fin(pathToWeights);
		for (int i = 0; i < input_size * output_size; i++) fin >> host_array_row[i];
		fin.close();

		for (int i = 0; i < input_size; i++) {
			for (int j = 0; j < output_size; j++) {
				host_array[i * output_size + j] = host_array_row[((j) * (input_size)) + (i)];
			}
		}
		cudaMalloc(&weights, output_size * input_size * sizeof(float));
		cudaMemcpy(weights, host_array, output_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
		delete[] host_array, host_array_row;
	};

	// считывание добавочных членов из файла
	void read_biases(std::string pathToWeights) {
		float* host_array = new float[output_size];

		std::ifstream fin(pathToWeights);
		for (int i = 0; i < output_size; i++) fin >> host_array[i];
		fin.close();

		cudaMalloc(&biases, output_size * sizeof(float));
		cudaMemcpy(biases, host_array, output_size * sizeof(float), cudaMemcpyHostToDevice);
		delete[] host_array;
	};

public:
	// конструкторы
	NN() {
		input_size = 0;
		output_size = 0;
		alpha = 1.0;
		beta = 1.0;
		activation_true = true;
	};

	NN(std::string pathToWeights, std::string pathToBiases, int inSize, int outSize, bool activation) {
		alpha = 1.0;
		beta = 1.0;
		input_size = inSize;
		output_size = outSize;
		read_weights(pathToWeights);
		read_biases(pathToBiases);
		activation_true = activation;

	};


	// линейный слой
	float* Linear(float* input) {
		cublasCreate(&handle);
		cublasSgemv(handle, CUBLAS_OP_N, output_size, input_size, &alpha, weights, output_size, input, 1, &beta, biases, 1);
		cublasDestroy(handle);
		if (activation_true) {
			nn_Sigmoid <<<1, output_size>>> (biases, output_size);
		}
		return biases;
	};

	// деструктор
	~NN() {
		if (weights != nullptr) cudaFree(weights);
		if (biases != nullptr) cudaFree(biases);
	};
};

///////////  MODEL   /////////////
class Net
{
private:
	float* array;
	int input_size, output_size;
	std::vector<NN> layers;

	// чтение input
	void read_input(std::string pathToWeights) {
		float* host_array = new float[input_size];

		std::ifstream fin(pathToWeights);
		for (int i = 0; i < input_size; i++) fin >> host_array[i];
		fin.close();


		cudaMalloc(&array, input_size * sizeof(float));
		cudaMemcpy(array, host_array, input_size * sizeof(float), cudaMemcpyHostToDevice);
		delete[] host_array;
	};

	// вывод
	void print_result(float* arr) {
		float* host_array = new float[output_size];
		cudaMemcpy(host_array, arr, output_size * sizeof(float), cudaMemcpyDeviceToHost);
		
		std::cout << "Result: " << std::endl;

		std::cout << host_array[0] << std::endl;

		//проврка
		float pattern = round(0.5757734179496765 * 100) / 100;
		host_array[0] = round(host_array[0] * 100) / 100;

		if (pattern == host_array[0]) std::cout << "IT'S RIGHT ANSWER!" << std::endl;
		else std::cout << "result(" << host_array[0] << ") and pattern(" << pattern << ")" << std::endl;

		delete[] host_array;
	};

public:
	// конструктор по умолчанию
	Net() {
		input_size = 1024;
		output_size = 1;
	};

	// запуск базовой сети
	void forward(std::string pathToFile) {
		read_input(pathToFile);
		NN layer1("weights1.bin", "biases1.bin", 1024, 256, true);
		array = layer1.Linear(array);
		NN layer2("weights2.bin", "biases2.bin", 256, 16, true);
		array = layer2.Linear(array);
		NN layer3("weights3.bin", "biases3.bin", 16, 1, true);
		array = layer3.Linear(array);

		print_result(array);
	}

	// деструктор
	~Net() {
		if (array != nullptr) cudaFree(array);//нулевой указатель 
	};
};

int main()
{
	Net model;
	model.forward("input.bin");
	return 0;
}
