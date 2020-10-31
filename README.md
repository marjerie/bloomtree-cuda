# bloomtree-cuda
CUDA Implementation of Parallel BloomTree - A Space-Efficient Approximate Representation for Graphs 

# Introduction
Parallel BloomTree is a space-efficient representation for graphs using bloom filters to store graphs in a compact
manner. MurmurHash3 has been used as the hash function in the bloom filter. The performance of the implementation is tested on the three algorithms
namely Breadth First Search, Greedy Vertex Coloring and Tarjan's Strongly Connected Components algorithm.

# Functions Available
The three major graph operations implemented using Bloom Tree are InsertEdge, GetNeighbours and IsEdge. 
```
InsertEdge(int num_vertices, int num_edges, int num_hashes, int num_bits, int *u, int *v, bool *bits)
GetNeighbours(int u, bool *bits, int num_vertices)
IsEdge(int u, int v, bool *bits, int num_vertices, int num_hashes, int num_bits)
```
The InsertEdge function parallelly all the edges present in the graph. GetNeighbours is used to obtain the neighbours of a given vertex. To check if an edge is present between two nodes, the function IsEdge is used.

# Run

The number of vertices, number of edges, number of bits or size of bloom filter, number of hash functions to be used and the edges present should be provided as input.

```
nvcc BloomTree.cu -o BloomTree
./BloomTree < *path-to-graph*
```

# References
1. [C++ Implementation of BloomTree](https://github.com/Kavitha-G/bloomtree) <br>
2. [CUDA Implementation for MurmurHash3](https://github.com/armon/cuda-hll) <br>
3. [Calculation of Lowest Common Ancestor](https://www.researchgate.net/publication/295186423_Properties_of_the_Lowest_Common_Ancestor_in_a_Complete_Binary_Tree)
