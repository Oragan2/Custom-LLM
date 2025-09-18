#include <iostream>
#include <vector>
#include "transformer.hpp"
#include <map>
#include <string>

int main() {
    int vocab_size = 30000;     // pretend vocab of 20 tokens
    int max_seq_len = 50000;    // max length of sequence
    int hidden_dim = 8;      // embedding dimension

    LLM model(vocab_size, max_seq_len, hidden_dim, 2);

    // test sequence of tokens
    model.forwardPass("This doesn't matter for the moment");

    return 0;
}
