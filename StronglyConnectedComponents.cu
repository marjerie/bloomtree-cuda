#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>

#define FORCE_INLINE

__device__ int Parent(int node);
__device__ int LeftChild(int node);
__device__ int RightChild(int node);
__device__ int Sibling(int node);
__device__ int calculate_lca(int u, int v);
__device__ void traversal(long int prev, long int lca, long int src, long int dest, int n, uint64_t *hash, bool *bit, int h, int m);
__device__ bool check_traversal_up(int prev, int lca, int src, int dest, int n, bool *mask);
__device__ bool check_traversal_down(int prev, int lca, int src, int dest, int n, bool *mask);
__device__ bool CheckBloom(long tid, uint64_t *hash_value, bool *bit, int h, int m);
__device__ void SetBloom(long tid, uint64_t *hash_value, bool *bit, int h, int m);

__device__ static inline FORCE_INLINE uint64_t rotl64 ( uint64_t x, int8_t r )
{
  return (x << r) | (x >> (64 - r));
}

#define ROTL64(x,y)	rotl64(x,y)
#define BIG_CONSTANT(x) (x##LLU)

#define getblock(p, i) (p[i])

__device__ static inline FORCE_INLINE uint64_t fmix64 ( uint64_t k )
{
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;

  return k;
}

__device__ void MurmurHash3_x64_128 ( int tid, char key[], const int len,
                           const uint32_t seed, void * out1 , void * out2)
{

  uint8_t data[16];
  for (int i=0; i<len; i++)
    data[i] = (uint8_t) key[i];
  const int nblocks = len / 16;
  int i;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  //----------
  // body

  uint64_t blocks[16];
  for (int i=0; i<len; i++)
    blocks[i] = (uint64_t) data[i];

  for(i = 0; i < nblocks; i++)
  {
    uint64_t k1 = blocks[i*2+0];
    uint64_t k2 = blocks[i*2+1];

    k1 *= c1; k1  = ROTL64(k1,31); k1 *= c2; h1 ^= k1;

    h1 = ROTL64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;

    k2 *= c2; k2  = ROTL64(k2,33); k2 *= c1; h2 ^= k2;

    h2 = ROTL64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
  }

  //----------
  // tail

  uint8_t tail[16];
  for (int i=0; i<len-nblocks*16; i++)
    tail[i] = (uint64_t) data[i+nblocks*16];

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch(len & 15)
  {
  case 15: k2 ^= (uint64_t)(tail[14]) << 48;
  case 14: k2 ^= (uint64_t)(tail[13]) << 40;
  case 13: k2 ^= (uint64_t)(tail[12]) << 32;
  case 12: k2 ^= (uint64_t)(tail[11]) << 24;
  case 11: k2 ^= (uint64_t)(tail[10]) << 16;
  case 10: k2 ^= (uint64_t)(tail[ 9]) << 8;
  case  9: k2 ^= (uint64_t)(tail[ 8]) << 0;
           k2 *= c2; k2  = ROTL64(k2,33); k2 *= c1; h2 ^= k2;

  case  8: k1 ^= (uint64_t)(tail[ 7]) << 56;
  case  7: k1 ^= (uint64_t)(tail[ 6]) << 48;
  case  6: k1 ^= (uint64_t)(tail[ 5]) << 40;
  case  5: k1 ^= (uint64_t)(tail[ 4]) << 32;
  case  4: k1 ^= (uint64_t)(tail[ 3]) << 24;
  case  3: k1 ^= (uint64_t)(tail[ 2]) << 16;
  case  2: k1 ^= (uint64_t)(tail[ 1]) << 8;
  case  1: k1 ^= (uint64_t)(tail[ 0]) << 0;
           k1 *= c1; k1  = ROTL64(k1,31); k1 *= c2; h1 ^= k1;
  };

  //----------
  // finalization

  h1 ^= len; h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  ((uint64_t*)out1)[0] = h1;
  ((uint64_t*)out2)[0] = h2;

}

__device__ inline uint64_t NthHash(uint8_t n, uint64_t hashA, uint64_t hashB, uint64_t filter_size) {
	return ((hashA + n * hashB) % filter_size);
}

__global__ void insert_edge(int *u, int *v, bool *bit, int n, int e, long int ful_vertices,long int valsperloop,int ii,int h, int m, uint64_t *hash)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y)+blockIdx.y*gridDim.x+blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
	int tid = blockNum*(blockDim.x*blockDim.y*blockDim.z)+threadNum;

	long int tid1 = ii*valsperloop + tid;

	if (tid1 < e){
		u[tid] = u[tid] + n - 1;
		v[tid] = v[tid] + n - 1;
		int src = u[tid];
		int dest = v[tid];

		if (!((u[tid] < ful_vertices && v[tid] < ful_vertices) || (u[tid] >= ful_vertices && v[tid] >= ful_vertices))) {
			if (u[tid] > v[tid]){
	 			int cur = Parent(u[tid]);
				SetBloom((cur*n+u[tid]-n+1) << 1,hash+2*tid*sizeof(uint64_t),bit,h,m);
				if (u[tid] == LeftChild(cur)){
					SetBloom((cur*n+v[tid]-n+1) << 1,hash+2*tid*sizeof(uint64_t),bit,h,m); 
				}
				else{
					SetBloom(((cur*n+v[tid]-n+1) << 1) + 1,hash+2*tid*sizeof(uint64_t),bit,h,m); 
				}
				u[tid] = cur;
			}	
		 	else{
	 			int cur = Parent(v[tid]);
				SetBloom((cur*n+v[tid]-n+1) << 1,hash+2*tid*sizeof(uint64_t),bit,h,m);
				if (v[tid] == LeftChild(cur)){
					SetBloom((cur*n+u[tid]-n+1) << 1,hash+2*tid*sizeof(uint64_t),bit,h,m); 
				}
				else{
					SetBloom(((cur*n+u[tid]-n+1) << 1) + 1,hash+2*tid*sizeof(uint64_t),bit,h,m); 
				}
				v[tid] = cur;
			}
		}
		
		__syncthreads();

		int lca = calculate_lca(u[tid], v[tid]);

		traversal(u[tid], lca, src, dest, n, hash+2*tid*sizeof(uint64_t),bit,h,m);
		traversal(v[tid], lca, dest, src, n, hash+2*tid*sizeof(uint64_t),bit,h,m);
	}
}

__global__ void SetBloom(bool *mask, uint64_t *hash_value, bool *bit, int h, int m, long int valsperloop, int ii)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y)+blockIdx.y*gridDim.x+blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
	int tid = blockNum*(blockDim.x*blockDim.y*blockDim.z)+threadNum;

	if (mask[tid] == 1){
		long tid1 = valsperloop*ii + tid;
		int i =0;
		int count=0; 	
		long num = tid1;
		char str[10];
		do{	
			count++;
			num /= 10;
		}while(num != 0); 
		num = tid1;
		do{	
			str[count-i-1] = num%10 + '0';
			num/=10;
			i++;
		}while(num !=0);	
		str[i] = 48+'\0';
		uint64_t len1 = (uint64_t) count;
		size_t len = (size_t) len1;
		MurmurHash3_x64_128(tid1, str, len, 0, (hash_value)+tid*2*sizeof(uint64_t), (hash_value)+(tid*2+1)*sizeof(uint64_t));	
	
		for (int i=0; i<h; i++){
			bit[NthHash(i,*((hash_value)+tid*2*sizeof(uint64_t)),*((hash_value)+(tid*2+1)*sizeof(uint64_t)),m)] = 1;
		}
	}
}

__global__ void set_mask(bool *mask, uint64_t *hash_value, bool *bit, int h, int m, int n, long int valsperloop, int ii)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y)+blockIdx.y*gridDim.x+blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
	int tid = blockNum*(blockDim.x*blockDim.y*blockDim.z)+threadNum;

	if ((valsperloop*ii + tid) < n){	
		long tid1 = valsperloop*ii + tid;
		mask[tid] = CheckBloom(tid1, (hash_value)+2*tid*sizeof(uint64_t), bit, h, m);
	}
}

__global__ void check_bloom(int *found, bool *bit, bool *mask, uint64_t *hash_value, int m, int h, int n)
{
	int val = blockIdx.x;
	int hash = threadIdx.x;
	uint64_t filter_size = (uint64_t) m;
	uint8_t hash_no = (uint8_t) hash;
	
	if (val < n && *found == 0){
		if (mask[val] == 1){
			 if (hash < h){
				if (bit[NthHash(hash_no,*(hash_value+2*val*sizeof(uint64_t)),*(hash_value+(2*val+1)*sizeof(uint64_t)), filter_size)] == 0){ 		
					atomicAdd(found, 1); 
				}
			}
		}
	}
}


__global__ void get_neighbours(int u, bool *neighs, bool *mask, int n, long int ful_vertices)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y)+blockIdx.y*gridDim.x+blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
	int tid = blockNum*(blockDim.x*blockDim.y*blockDim.z)+threadNum;

	long v=tid;

	*(neighs+(tid)*sizeof(bool)) = 0;

	if (tid < n && !(u==tid) ){

		u = u + n - 1;
		v = v + n - 1;
		long src = u;
		long dest = v;

		if (!((u < ful_vertices && v < ful_vertices) || (u >= ful_vertices && v >= ful_vertices))) {
			if (u > v){
	 			long cur = Parent(u);
				if ((mask[(cur*n+u-n+1) << 1] == 0)) return;
				u = cur;
			}	
		 	else{
	 			long cur = Parent(v);
				if (v == LeftChild(cur))
				{
					if ((mask[(cur*n+u-n+1) << 1] == 0)) 
						return;
				}
				else
				{
					if ((mask[((cur*n+u-n+1) << 1) + 1] == 0)) 
						return;
				} 
				v = cur;
			}
		}
	
		//__syncthreads();

		int lca = calculate_lca(u, v);
		if (check_traversal_up(u, lca, src, dest, n, mask)) 
		{
			if(check_traversal_down(v, lca, dest, src, n, mask))
			{
				*(neighs+(dest-n+1)*sizeof(bool)) = 1;
			}
		}
	}
}


__device__ int Parent(int node)
{
	return (((node + 1) >> 1) - 1); 
}

__device__ int LeftChild(int node)
{
	return (((node + 1) << 1) - 1); 
}

__device__ int RightChild(int node)
{
	return ((node + 1) << 1); 
}

__device__ int Sibling(int node)
{
	return (((node + 1) ^ 1) - 1); 
}

__device__ int calculate_lca(int u, int v)
{	
	int val1 = 0;
	int val2 = 0;	
	int i = 1;

	do{
		float pow_val = 1 << i;
		val1 = floor((u+1)/pow_val);
		val2 = floor((v+1)/pow_val);
		i++;
	} while(val1 != val2);
	return (val1 - 1);
}

__device__ void traversal(long int prev, long int lca, long int src, long int dest, int n, uint64_t *hash, bool *bit, int h, int m)
{
	int cur = Parent(prev);
	while (cur != lca){
		SetBloom((cur*n+src-n+1) << 1,hash,bit,h,m);
		if (prev == LeftChild(cur)){
			SetBloom((cur*n+dest-n+1) << 1,hash,bit,h,m); 
		}
		else{
			SetBloom(((cur*n+dest-n+1) << 1) + 1,hash,bit,h,m); 
		} 
		prev = cur;
		cur = Parent(cur);
	}
	SetBloom(((cur*n+src-n+1) << 1) + 1,hash,bit,h,m);	
}

__device__ bool check_traversal_up(int prev, int lca, int src, int dest, int n, bool *mask)  
{
	int cur = Parent(prev);
	while (cur != lca){
		if ((mask[(cur*n+src-n+1) << 1]) == 0)  return false;
		prev = cur;
		cur = Parent(cur);
	}
	if ((mask[((cur*n+src-n+1) << 1) + 1]) == 0)  return false;
	return true;	
}

__device__ bool check_traversal_down(int prev, int lca, int src, int dest, int n, bool *mask) 
{
	int cur = Parent(prev);
	while (cur != lca){
		if (prev == LeftChild(cur))
		{
			if ((mask[(cur*n+dest-n+1) << 1]) == 0)  
				return false;
		}
		else
		{
			if ((mask[((cur*n+dest-n+1) << 1) + 1]) == 0) 
				return false;
		}
		prev = cur;
		cur = Parent(cur);
	}
	return true;	
}

__device__ void SetBloom(long tid, uint64_t *hash_value, bool *bit, int h, int m)
{
	int i =0;
	int count=0; 	
	long num = tid;
	char str[11];
	do{	
		count++;
		num /= 10;
	}while(num != 0); 
	num = tid;
	do{	
		str[count-i-1] = num%10 + '0';
		num/=10;
		i++;
	}while(num !=0);	
	str[i] = 48+'\0';
	uint64_t len1 = (uint64_t) count;
	size_t len = (size_t) len1;
	MurmurHash3_x64_128(tid, str, len, 0, (hash_value), (hash_value)+sizeof(uint64_t));

	for (int i=0; i<h; i++){
		bit[NthHash(i,*(hash_value),*((hash_value)+sizeof(uint64_t)),m)] = 1;
	}
}

__device__ bool CheckBloom(long tid, uint64_t *hash_value, bool *bit, int h, int m)
{
	int i =0;
	int count=0; 	
	long num = tid;
	char str[10];
	do{	
		count++;
		num /= 10;
	}while(num != 0); 
	num = tid;
	do{	
		str[count-i-1] = num%10 + '0';
		num/=10;
		i++;
	}while(num !=0);	
	str[i] = 48+'\0';
	uint64_t len1 = (uint64_t) count;
	size_t len = (size_t) len1;
	MurmurHash3_x64_128(tid, str, len, 0, (hash_value), (hash_value)+sizeof(uint64_t));

	for (int i=0; i<h; i++){
		if (bit[NthHash(i,*(hash_value),*((hash_value)+sizeof(uint64_t)),m)] == 0){
			return false;
		}
	}
	return true;
}

void InsertEdge(int num_vertices, int num_edges, int num_hashes, int num_bits, int *h_u, int *h_v, bool *h_bits)
{

	cudaError_t err = cudaSuccess;
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	size_t size = num_edges * sizeof(int);

	int num_ful_levels = floor( log2((double) (2*num_vertices - 1)));
	long int ful_vertices = pow((int) 2,(int) num_ful_levels) - 1;

	size_t size_bits = num_bits * sizeof(bool);
	bool *d_bits = NULL;
        err=cudaMalloc((void **)&d_bits, size_bits);
	cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to allocate vector bits (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	cudaMemset(d_bits, 0, size_bits);

	int *d_u = NULL, *d_v = NULL; 
	err=cudaMalloc((void **)&d_u, size);
	cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to allocate vector u (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}
	err=cudaMalloc((void **)&d_v, size);
	cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to allocate vector v (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}
	err=cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to copy u (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}
	err=cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to copy v (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	clock_t ti;

	cudaEventRecord(start1);
	ti=clock();

	dim3 tpb1(32,32,1);
   	dim3 bpg1(8,2,1);

	long int valsperloop = 1 << 14;

	int N = ceil(num_edges/valsperloop)+1;

	uint64_t *d_hash_value = NULL;
	size_t size_hash = 2*num_edges*sizeof(uint64_t);
	cudaMalloc((void **)&d_hash_value, size_hash);

	for (int i=0; i<N; i++){
		insert_edge<<<bpg1,tpb1>>>(d_u+i*valsperloop, d_v+i*valsperloop, d_bits, num_vertices, num_edges, ful_vertices, valsperloop, i, num_hashes, num_bits, d_hash_value);

		/*cudaDeviceSynchronize();
	
		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
		    fprintf(stderr, "Failed to launch insert_edge kernel %d (error code %s)!\n",i, cudaGetErrorString(err));
		    exit(EXIT_FAILURE);
		}*/
	}

	cudaMemcpy(h_bits, d_bits, size_bits, cudaMemcpyDeviceToHost);

	ti=clock() -ti;
	cudaEventRecord(stop1);

	cudaEventSynchronize(stop1);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start1, stop1);

	printf("%.5f\n", float(ti) / CLOCKS_PER_SEC);

	printf("time taken is %.5f\n", milliseconds);

	int value=0;

	for (long i =0; i<num_bits; i++){
		if (h_bits[i] == 1) value++;
	}

	//printf("value is %d\n",value);

	cudaFree(d_u);
	cudaFree(d_v);
	cudaFree(d_hash_value);
	cudaFree(d_bits);
}

const int Num = 2e5 + 5;
int WH = -1, BL = 1, num_sccs;
int dfs_num[Num], dfs_low[Num], dfscounter;

int stackn=4040, top = -1, *stack;
void push(int x){
	if(top==stackn) return;
	stack[++top]=x;
}

int pop(){
	if(top==-1) return -1;
	return stack[top--];
}

int peek(){
	if(top==-1) return -1;
	return stack[top];
}

int isempty() {
   if(top == -1) return 1;
   else return 0;
}

int vectorn=4040, total =0,*vector;

int vector_search(int low, int high, int key)
{
    if (high < low)
        return -1;
    int mid = (low + high) / 2;
    if (key == vector[mid])
        return mid;
    if (key > vector[mid])
        return vector_search((mid + 1), high, key);
    return vector_search(low, (mid - 1), key);
}

void vector_add(int x){
    int i;
    if (total == 0) vector[total]=x;
    else{
	 for (i = total - 1; (i >= 0 && vector[i] > x); i--)
	    vector[i + 1] = vector[i];
	 vector[i + 1] = x;
    }
    total++;
}

void vector_del(int x)
{
    int pos = vector_search(0, total - 1, x);
    int i;
    for (i = pos; i < total - 1; i++)
        vector[i] = vector[i + 1];
    total--;
}

bool vector_find(int x)
{
    int pos = vector_search(0, total - 1, x);
    if (pos == -1) return false;
    else return true;
}

void SCC(int u, bool *d_mask, int num_vertices, long int ful_vertices) 
{
	cudaError_t err = cudaSuccess;

	dfs_low[u] = dfs_num[u] = dfscounter++;
	push(u);
	vector_add(u);

	bool *d_neighs = NULL;
	size_t size_neighs = (num_vertices)*sizeof(bool);
	cudaMalloc((void **)&d_neighs, size_neighs);
	bool *h_neighs = (bool *)malloc(size_neighs);

	dim3 tpb2(32,32,1);
    dim3 bpg2(2,2,1);

	get_neighbours<<<bpg2,tpb2>>>(u, d_neighs, d_mask, num_vertices, ful_vertices);

	cudaDeviceSynchronize();

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to launch get_neighbours kernel (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}
	cudaMemcpy(h_neighs, d_neighs, size_neighs, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to copy (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

	for (int i = 0; i < num_vertices; ++i) {
		if (h_neighs[i] == 1){	
			if (dfs_num[i] == WH) {
				SCC(i,d_mask,num_vertices,ful_vertices);
			}
			if (vector_find(i)) {
				dfs_low[u] = min(dfs_low[u], dfs_low[i]);
			}
		}
	}
	
	if (dfs_low[u] == dfs_num[u]) { 
		num_sccs++;
		while (!isempty() && peek() != u){
			vector_del(peek());
			pop();
		}
		vector_del(peek());
		pop();
	}

	cudaFree(d_neighs);
	free(h_neighs);
}

int main ()
{

	cudaError_t err = cudaSuccess;
	int num_vertices, num_edges, num_hashes, num_bits;
	scanf("%d",&num_vertices);
	scanf("%d",&num_edges);
	scanf("%d",&num_bits);
	scanf("%d",&num_hashes);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	size_t size = num_edges * sizeof(int);
	int num_vals = 2*num_vertices*(num_vertices-1);

	int *h_u = (int *)malloc(size);
	int *h_v = (int *)malloc(size);
	
	for (int i =0; i<num_edges; i++)
	{
		scanf("%d",&h_u[i]);
		scanf("%d",&h_v[i]);
	}

	size_t size_bits = num_bits * sizeof(bool);
	bool *h_bits = (bool *)malloc(size_bits);

	InsertEdge(num_vertices, num_edges, num_hashes, num_bits, h_u, h_v, h_bits);

	uint64_t *d_hash_value = NULL;
	size_t size_hash = 2*num_vals*sizeof(uint64_t);
	cudaMalloc((void **)&d_hash_value, size_hash);

	size_t size_mask = num_vals * sizeof(bool);
	bool *h_mask = (bool *)malloc(size_mask);
	bool *d_mask = NULL;
    cudaMalloc((void **)&d_mask, size_mask);

	bool *d_bits = NULL;
    cudaMalloc((void **)&d_bits, size_bits);
	cudaMemcpy(d_bits, h_bits, size_bits, cudaMemcpyHostToDevice);
	
	int num_ful_levels = floor( log2((double) (2*num_vertices - 1)));
	long int ful_vertices = pow((int) 2,(int) num_ful_levels) - 1;

	dim3 tpb2(32,32,1);
    dim3 bpg2(2,2,1);

	dim3 tpb3(32,32,1);
    dim3 bpg3(4,4,1);
	long int valsperloop = 1 << 14;

	int N = ceil(num_vals/valsperloop)+1;

	clock_t ti;

	cudaEventRecord(start);

	ti=clock();

	for (int i = 0; i < num_vertices; ++i) {
		dfs_num[i]=WH;
	}

	for (int i=0; i<N; i++){
		set_mask<<<bpg3,tpb3>>>(d_mask+i*valsperloop,d_hash_value+i*2*valsperloop,d_bits,num_hashes,num_bits,num_vals,valsperloop,i);

        	cudaDeviceSynchronize();

		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
		    fprintf(stderr, "Failed to launch set_mask kernel %d (error code %s)!\n",i, cudaGetErrorString(err));
		    exit(EXIT_FAILURE);
		}
	}

	stack = (int*)malloc(stackn*sizeof(int));
	vector = (int*)malloc(vectorn*sizeof(int));

	dfscounter = 0;
	num_sccs = 0;
	for (int i = 0; i < num_vertices; ++i) {
		if (dfs_num[i] == WH) {
			SCC(i,d_mask,num_vertices,ful_vertices);
		}
	}

	ti=clock()-ti;

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("number of SCCs: %d\n",num_sccs);

	printf("%.5f\n", float(ti) / CLOCKS_PER_SEC);

	printf("time taken is %.5f\n", milliseconds);

	cudaFree(d_hash_value);
	cudaFree(d_bits);
	cudaFree(d_mask);

	free(h_u);
	free(h_v);
	free(h_bits);
	
	cudaDeviceReset();
	return 0;
}
