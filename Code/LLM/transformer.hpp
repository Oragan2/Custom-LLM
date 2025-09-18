/**
 * @file transformer.hpp
 * @brief Header file defining core structures and functions for a simple Transformer-based LLM.
 */

/// @struct MultiHeadAttention
/// @brief Implements multi-head self-attention mechanism.
struct MultiHeadAttention{
    /**
     * @brief Number of attention heads.
     */
    int num_heads;

    /**
     * @brief Total hidden dimension of the model.
     */
    int hidden_dim;

    /**
     * @brief Dimension of each attention head.
     */
    int head_dim;

    /**
     * @brief Weight matrices for query, key, value, and output projections.
     */
    std::vector<std::vector<float>> W_Q, W_K, W_V, W_O;

    /**
     * @brief Constructs a MultiHeadAttention layer.
     * @param h Number of attention heads.
     * @param d Hidden dimension.
     */
    MultiHeadAttention(int h, int d);

    /**
     * @brief Forward pass for multi-head attention.
     * @param X Input tensor of shape (sequence_length, hidden_dim).
     * @return Output tensor after attention.
     */
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& X);
};

/// @class LLM
/// @brief Represents a simple Transformer-based language model.
class LLM
{
public:
    /**
     * @brief Constructs the LLM model.
     * @param vocab_size Size of the vocabulary.
     * @param max_seq_len Maximum sequence length.
     * @param hidden_dim Hidden dimension size.
     * @param num_head Number of attention heads.
     */
    LLM(int vocab_size, int max_seq_len, int hidden_dim, int num_head);

    /**
     * @brief Do a forward pass of the LLM
     * @param text The text that will be passed through the LLM
     * @return Return the computed string
     */
    std::vector<std::vector<float>> forwardPass(std::string text);

private:
    /**
     * @brief Embedding matrix for tokens.
     */
    std::vector<std::vector<float>> embeding;

    /**
     * @brief Positional encoding matrix.
     */
    std::vector<std::vector<float>> positionalEncoding;

    /**
     * @brief Multi-head attention transformer layer.
     */
    MultiHeadAttention transformer;

    /**
     * @brief Generates sinusoidal positional encodings.
     * @param seq_len Sequence length.
     * @param dim Embedding dimension.
     * @return Sinusoidal positional encoding matrix.
     */
    std::vector<std::vector<float>> sinusoidalEncoding(int seq_len, int dim);
    
    /**
     * @brief Computes token embeddings for a sequence of tokens.
     * @param tokens Input token indices.
     * @return Embedded representation of tokens.
     */
    std::vector<std::vector<float>> tokenEmbeding(std::vector<int>& tokens);
};

/**
 * @brief Initializes a matrix with random values.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Initialized matrix.
 */
std::vector<std::vector<float>> initialize_matrix(int rows, int cols);

/**
 * @brief Performs matrix multiplication.
 * @param A Left matrix.
 * @param B Right matrix.
 * @return Result of A * B.
 */
std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B);

/**
 * @brief Applies softmax function row-wise to a matrix.
 * @param mat Input matrix.
 * @return Matrix after softmax.
 */
std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& mat);

/**
 * @brief Adds two matrices element-wise.
 * @param A First matrix.
 * @param B Second matrix.
 * @return Resulting matrix.
 */
std::vector<std::vector<float>> add(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B);

/**
 * @brief Transposes a matrix.
 * @param mat Input matrix.
 * @return Transposed matrix.
 */
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat);

/**
 * @brief Extracts a submatrix (slice) from a matrix.
 * @param mat Input matrix.
 * @param row_start Starting row index.
 * @param row_end Ending row index (exclusive).
 * @param col_start Starting column index.
 * @param col_end Ending column index (exclusive).
 * @return Sliced submatrix.
 */
std::vector<std::vector<float>> slice(const std::vector<std::vector<float>>& mat, int row_start, int row_end, int col_start, int col_end);

/**
 * @brief Applies a function element-wise to a matrix.
 * @param mat Input matrix.
 * @param func Function pointer to apply.
 * @return Matrix after function application.
 */
std::vector<std::vector<float>> apply_function(const std::vector<std::vector<float>>& mat, float (*func)(float));