#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <map>
#include <string>
#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>

// Global variables and constants

std::vector<uint64_t> merges; // Store the merges
std::map<std::string, int> vocab; // Store the tokens and their ids
std::map<int, std::string> i_vocab; // Store the ids and their token
int batch_size = 1024 * 1024 * 4; // 4MB batch
std::vector<std::vector<uint32_t>> corpus_batches; // Store the batches of the corpus
// GPU variable
uint32_t* d_corpus;
uint32_t* d_hash_t;
uint64_t* d_pairs_table;
uint32_t* new_corpus;
uint32_t* place;
const size_t table_size = 200000000; // number of slots
int new_token_id = 0;
struct is_one {
    __host__ __device__
    bool operator()(const int x) const {
        return x == 1;
    }
};


// GPU functions declarations

__device__ inline uint64_t hash64(uint64_t key) {
	key ^= key >> 33;
	key *= 0xff51afd7ed558ccdULL;
	key ^= key >> 33;
	key *= 0xc4ceb9fe1a85ec53ULL;
	key ^= key >> 33;
	return key;
}

__global__ void GeneratePairs(uint32_t* corpus, uint32_t* hashs, uint64_t* hash_p, size_t N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	const size_t table_size = 200000000; // number of slots

	for (size_t i = idx; i < N - 1; i += stride) {
		uint64_t pair = ((uint64_t)corpus[i] << 32) | (uint64_t)corpus[i + 1];
		uint64_t hash_idx = hash64(pair) % table_size;
		uint64_t old = atomicAdd(&hashs[hash_idx], 1);

		if (old == 0) {
			hash_p[hash_idx] = pair;
		}
	}
}

__global__ void findBest(uint32_t* corpus, uint64_t best_pair, uint32_t* place, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (size_t i = idx; i < N - 1; i += stride) {
		uint64_t pair = ((uint64_t)corpus[i] << 32) | (uint64_t)corpus[i + 1];
		if (pair == best_pair) {
			place[i] = 1;
		}
		else {
			place[i] = 0;
		}
	}
}

__global__ void updateHash() {
	
}

// CPU helper functions declarations

void write() {
	std::ofstream merge_file("merges.txt", std::ios::app);
	for (auto& p : merges) {
		merge_file << (int)(uint32_t)(p >> 32) << " " << (int)(uint32_t)(p & 0xFFFFFFFF) << "\n";
	}
	merge_file.close();
	std::ofstream vocab_file("vocab.json", std::ios::app);
	for (auto it = vocab.begin(); it != vocab.end(); ) {
		vocab_file << "  \"" << it->first << "\": " << it->second;
		if (++it != vocab.end()) vocab_file << ",";
		vocab_file << "\n";
	}
	vocab_file.close();
}

void hash_table() {
	/// Create the pairs and their counts
	cudaEvent_t start, stop;
	float milliseconds = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start timing
	cudaEventRecord(start);

	for (auto& c : corpus_batches) {
		// Copy batch to device
		size_t N = c.size();
		cudaMemcpy(d_corpus, c.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

		// Define kernel launch parameters
		int blockSize = 256;
		int numBlocks = (N + blockSize - 1) / blockSize;

		// Launch kernel to generate pairs
		GeneratePairs << <numBlocks, blockSize >> > (d_corpus, d_hash_t, d_pairs_table, N);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
			std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;

		// Wait for GPU to finish
		cudaDeviceSynchronize();
	}

	// Stop timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time to build hash table: " << milliseconds << " ms" << std::endl;
	
	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

uint64_t maxi() {
	cudaEvent_t start, stop;
	float milliseconds = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);

	auto max_it = thrust::max_element(thrust::device, d_hash_t, d_hash_t + table_size);
	int max_idx = max_it - d_hash_t;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time to find max_element: " << milliseconds << " ms" << std::endl;

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy hash table back to host
	uint64_t host_most_frequent_pair;
	cudaMemcpy(&host_most_frequent_pair, &d_pairs_table[max_idx], sizeof(uint64_t), cudaMemcpyDeviceToHost);
	vocab.emplace(i_vocab[(int)(uint32_t)host_most_frequent_pair >> 32]+i_vocab[(int)(uint32_t)host_most_frequent_pair & 0xFFFFFFFF], new_token_id++);
	return host_most_frequent_pair;
}

void merge(uint64_t most_frequent_pair) {
	cudaEvent_t start, stop;
	float milliseconds = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start timing
	cudaEventRecord(start);
	for (auto& c : corpus_batches) {
			// Copy batch to device
			size_t N = c.size();
			cudaMemcpy(d_corpus, c.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

			// Define kernel launch parameters
			int blockSize = 256;
			int numBlocks = (N + blockSize - 1) / blockSize;

			// Launch kernel to generate pairs
			findBest << <numBlocks, blockSize >> > (d_corpus, most_frequent_pair, place , N);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess)
				std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;

			// Wait for GPU to finish
			cudaDeviceSynchronize();
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time to build hash table: " << milliseconds << " ms" << std::endl;
	
	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

// main
int main() {
	// Load the file
	std::ifstream text8("text8", std::ios::binary);
	if (!text8) {
		std::cerr << "Error opening text8 file" << std::endl;
		return 1;
	}
	std::vector<char> raw((std::istreambuf_iterator<char>(text8)), {});
	text8.close();
	size_t corpus_size = raw.size();
	std::cout << "Read " << corpus_size << " bytes\n";
	// Split the file into batches
	std::vector<std::vector<char>> batches;
	std::cout << "Creating batches of size " << batch_size << " bytes\n";
	for (size_t i = 0; i < corpus_size; i += batch_size) {
		size_t end = std::min(i + batch_size, corpus_size);
		batches.emplace_back(raw.begin() + i, raw.begin() + end); // Slice the raw data into batches
	}
	// Turn the ASSCI into uint8
	for (auto& batch : batches) {
		std::vector<uint32_t> corpus(batch.size());
		for (size_t i = 0; i < batch.size(); i++) {
			corpus[i] = (uint32_t)batch[i];
		}
		corpus_batches.push_back(std::move(corpus));
	}

	std::cout << "Created " << corpus_batches.size() << " batches\n";

	int vocab_size = 500; // Desired vocab size

	// Initialize the vocab with single characters

	vocab.emplace("<EOS>", new_token_id++); // End of sequence token
	vocab.emplace("<UNK>", new_token_id++); // Unknown token
	vocab.emplace("<CODE>", new_token_id++); // Code token
	vocab.emplace("<FR>", new_token_id++); // French token
	vocab.emplace("<EN>", new_token_id++); // English token
	for (int i = 32; i < 256; i++) {
		std::string s(1, (char)i);
		vocab.emplace(s, i);
		i_vocab.emplace(i, s);
	}
	new_token_id = 257; // Reset new_token_id to 257 for me to have token 0-32 for special ones

	cudaMalloc(&d_corpus, batch_size * sizeof(uint32_t));
	cudaMalloc(&d_hash_t, table_size * sizeof(uint32_t));
	cudaMalloc(&d_pairs_table, table_size * sizeof(uint64_t));
	cudaMalloc(&new_corpus, batch_size * sizeof(uint32_t));
	cudaMalloc(&place, batch_size * sizeof(uint32_t));

	hash_table();	

	while(new_token_id < vocab_size) {
		uint64_t most_frequent_pair = maxi();
		merge(most_frequent_pair);
		write();
	}

	cudaFree(&d_corpus);
	cudaFree(&d_hash_t);
	cudaFree(&d_pairs_table);
	cudaFree(&new_corpus);
	cudaFree(&place);
	return 0;
}