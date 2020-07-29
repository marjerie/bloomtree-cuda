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
__device__ void traversal(long int prev, long int lca, long int src, long int dest, bool *mask, int n);
__device__ bool check_traversal_up(int prev, int lca, int src, int dest, int n, uint64_t *hash_value, bool *bit, int h, int m);
__device__ bool check_traversal_down(int prev, int lca, int src, int dest, int n, uint64_t *hash_value, bool *bit, int h, int m);
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
  //const uint8_t * data = (const uint8_t*)key;
  //printf("tid = %d, %c, %c, %c, %c, %c, %c, %c, %c, %c\n",tid,key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7],key[8]);

  //if (key[0]== 48) {printf("h "); key[0] = 48+'\0';}

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

  //const uint64_t * blocks = (const uint64_t *)(data);

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

  //const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

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
	//printf("%" PRIu64 " and %" PRIu64 " %u \n",hashA,hashB,((hashA + n * hashB) % filter_size));
	//printf("%u\n",n);
	//printf ("nth %u\n",((hashA + n * hashB) % filter_size));
	return ((hashA + n * hashB) % filter_size);
}


__global__ void init_mask(bool *mask, int n)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y)+blockIdx.y*gridDim.x+blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
	int tid = blockNum*(blockDim.x*blockDim.y*blockDim.z)+threadNum;

	if (tid < n)
		mask[tid] = 0;
}

__global__ void init_bits(bool *array, int n)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	if (tid < n){
		array[tid] = 0;
	}
}

__global__ void get_mask(int *u, int *v, bool *mask, int n, int e, long int ful_vertices,long int valsperloop,int ii)
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
				mask[(cur*n+u[tid]-n+1) << 1] = 1;
				if (u[tid] == LeftChild(cur)){
					mask[(cur*n+v[tid]-n+1) << 1] = 1; 
				}
				else{
					mask[((cur*n+v[tid]-n+1) << 1) + 1] = 1; 
				}
				u[tid] = cur;
			}	
		 	else{
	 			int cur = Parent(v[tid]);
				mask[(cur*n+v[tid]-n+1) << 1] = 1;
				if (v[tid] == LeftChild(cur)){
					mask[(cur*n+u[tid]-n+1) << 1] = 1; 
				}
				else{
					mask[((cur*n+u[tid]-n+1) << 1) + 1] = 1; 
				}
				v[tid] = cur;
			}
		}
		
		__syncthreads();

		int lca = calculate_lca(u[tid], v[tid]);

		traversal(u[tid], lca, src, dest, mask, n);
		traversal(v[tid], lca, dest, src, mask, n);
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
					atomicAdd(found, 1); //printf("here");
				}
			}
		}
	}
}


__global__ void get_neighbours(int u, bool *neighs, uint64_t *hash_value, bool *bit, int m, int h, int n, long int ful_vertices)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y)+blockIdx.y*gridDim.x+blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
	int tid = blockNum*(blockDim.x*blockDim.y*blockDim.z)+threadNum;

	long v=tid;

	if (tid < n && !(u==tid) ){

		u = u + n - 1;
		v = v + n - 1;
		long src = u;
		long dest = v;

		if (!((u < ful_vertices && v < ful_vertices) || (u >= ful_vertices && v >= ful_vertices))) {
			if (u > v){
	 			long cur = Parent(u);
				if (!(CheckBloom(((cur*n+u-n+1) << 1),hash_value+2*tid*sizeof(uint64_t),bit,h,m))) return;
				u = cur;
			}	
		 	else{
	 			long cur = Parent(v);
				if (v == LeftChild(cur))
				{
					if (!(CheckBloom(((cur*n+u-n+1) << 1),hash_value+2*tid*sizeof(uint64_t),bit,h,m))) 
						return;
				}
				else
				{
					if (!(CheckBloom((((cur*n+u-n+1) << 1) + 1),hash_value+2*tid*sizeof(uint64_t),bit,h,m))) 
						return;
				}
				v = cur;
			}
		}
	
		__syncthreads();

		int lca = calculate_lca(u, v);
		if (check_traversal_up(u, lca, src, dest, n, hash_value+2*tid*sizeof(uint64_t), bit, h, m)) 
		{
			if(check_traversal_down(v, lca, dest, src, n, hash_value+2*tid*sizeof(uint64_t), bit, h, m))
			{
				*(neighs+(dest-n+1)*sizeof(bool)) = 1;
			}
		}
	}
}

__global__ void print_hash(uint64_t *hash_value, int n)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y)+blockIdx.y*gridDim.x+blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
	int tid = blockNum*(blockDim.x*blockDim.y*blockDim.z)+threadNum;

	if (tid < 4*n*(n-1)){
		//printf("%u\n",*(hash_value+tid*sizeof(uint8_t)));
		printf("%" PRIu64 " %d\n",*(hash_value+tid*sizeof(uint64_t)),tid);
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

__device__ void traversal(long int prev, long int lca, long int src, long int dest, bool *mask, int n)
{
	int cur = Parent(prev);
	while (cur != lca){
		mask[(cur*n+src-n+1) << 1] = 1;
		if (prev == LeftChild(cur)){
			mask[(cur*n+dest-n+1) << 1] = 1; 
		}
		else{
			mask[((cur*n+dest-n+1) << 1) + 1] = 1; 
		} 
		prev = cur;
		cur = Parent(cur);
	}
	mask[((cur*n+src-n+1) << 1) + 1] = 1;	
}

__device__ bool check_traversal_up(int prev, int lca, int src, int dest, int n, uint64_t *hash_value, bool *bit, int h, int m)  
{
	int cur = Parent(prev);
	while (cur != lca){
		if (!(CheckBloom(((cur*n+src-n+1) << 1),hash_value,bit,h,m)))  return false;
		prev = cur;
		cur = Parent(cur);
	}
	if (!(CheckBloom((((cur*n+src-n+1) << 1) + 1),hash_value,bit,h,m)))  return false;
	return true;	
}

__device__ bool check_traversal_down(int prev, int lca, int src, int dest, int n, uint64_t *hash_value, bool *bit, int h, int m) 
{
	int cur = Parent(prev);
	while (cur != lca){
		if (prev == LeftChild(cur))
		{
			if (!(CheckBloom(((cur*n+dest-n+1) << 1),hash_value,bit,h,m)))  
				return false;
		}
		else
		{
			if (!(CheckBloom((((cur*n+dest-n+1) << 1) + 1),hash_value,bit,h,m))) 
				return false;
		}
		prev = cur;
		cur = Parent(cur);
	}
	return true;	
}


__device__ bool CheckBloom(long tid, uint64_t *hash_value, bool *bit, int h, int m)
{
	//printf("%ld ",tid);
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
	size_t size = num_edges * sizeof(int);
	int num_vals = 2*num_vertices*(num_vertices-1);

	int *d_u = NULL, *d_v = NULL; 
        cudaMalloc((void **)&d_u, size);
	cudaMalloc((void **)&d_v, size);
	cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);

	int num_ful_levels = floor( log2((double) (2*num_vertices - 1)));
	long int ful_vertices = pow((int) 2,(int) num_ful_levels) - 1;

	size_t size_mask = num_vals * sizeof(bool);
	//bool *h_mask = (bool *)malloc(size_mask);
	bool *d_mask = NULL;
        cudaMalloc((void **)&d_mask, size_mask);

	//dim3 tpb(32,32,1);
	//dim3 bpg(ceil(num_vertices/32),ceil((num_vertices-1)/32),2);

	//init_mask<<<bpg,tpb>>>(d_mask,num_vertices);

	cudaMemset(d_mask, 0, size_mask);

	size_t size_bits = num_bits * sizeof(bool);
	//bool *h_bits = (bool *)malloc(size_bits);
	bool *d_bits = NULL;
        cudaMalloc((void **)&d_bits, size_bits);

	//init_mask<<<2,num_bits/2>>>(d_bits,num_bits);
	cudaMemset(d_bits, 0, size_bits);

	//int M = num_edges/4096;
	//printf("M is %d\n",M);

	dim3 tpb1(8,8,1);
        dim3 bpg1(4,4,1);

	long int valsperloop = 1 << 10;

	int N = ceil(num_edges/valsperloop)+1;

	printf("N is %d\n",N);

	for (int i=0; i<N; i++){
		get_mask<<<bpg1,tpb1>>>(d_u+i*valsperloop,d_v+i*valsperloop,d_mask,num_vertices,num_edges,ful_vertices,valsperloop,i);

		cudaDeviceSynchronize();
	
		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
		    fprintf(stderr, "Failed to launch get_mask kernel %d (error code %s)!\n",i, cudaGetErrorString(err));
		    exit(EXIT_FAILURE);
		}
	}

	//get_mask<<<bpg1,tpb1>>>(d_u,d_v,d_mask,num_vertices,num_edges,ful_vertices);

	uint64_t *d_hash_value = NULL;
	size_t size_hash = 2*num_vals*sizeof(uint64_t);
	cudaMalloc((void **)&d_hash_value, size_hash);

	//cudaMemcpy(h_mask, d_mask, size_mask, cudaMemcpyDeviceToHost);
	
	cudaFree(d_u);
	cudaFree(d_v);

	dim3 tpb2(8,4,1);
        dim3 bpg2(2,2,2);

	valsperloop = 1 << 8;

	N = ceil(num_vals/valsperloop)+1;

	printf("N is %d\n",N);

	for (int i=0; i<N; i++){
		SetBloom<<<bpg2,tpb2>>>(d_mask+i*valsperloop,d_hash_value+i*2*valsperloop,d_bits,num_hashes,num_bits,valsperloop,i);

		cudaDeviceSynchronize();
	
		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
		    fprintf(stderr, "Failed to launch SetBloom kernel %d (error code %s)!\n",i, cudaGetErrorString(err));
		    exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(h_bits, d_bits, size_bits, cudaMemcpyDeviceToHost);

	int value=0;

	for (long i =0; i<num_bits; i++){
		if (h_bits[i] == 1) value++;
	}

	printf("value is %d\n",value);
	cudaFree(d_hash_value);
	cudaFree(d_mask);
	cudaFree(d_bits);
}

/*bool IsEdge(int u, int v, int num_vertices, int num_hashes, int num_bits, bool *h_bits)
{
	size_t size = sizeof(int);
	int *d_u = NULL, *d_v = NULL;
	int *h_eu = (int *)malloc(size);
	int *h_ev = (int *)malloc(size); 
        cudaMalloc((void **)&d_u, size);
	cudaMalloc((void **)&d_v, size);
	h_eu[0] = u;
	h_ev[0] = v;
	cudaMemcpy(d_u,h_eu,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_v,h_ev,size,cudaMemcpyHostToDevice);

	int num_vals = 2*num_vertices*(num_vertices-1);
	size_t size_mask = num_vals * sizeof(bool);
	bool *d_mask = NULL;
        cudaMalloc((void **)&d_mask, size_mask);
	dim3 tpb(num_vertices,(2*(num_vertices-1)),1);
	dim3 bpg(1,1,1);
	init_mask<<<bpg,tpb>>>(d_mask,num_vertices);

	int tpb1 = 1;
        int bpg1 = 1;
	int num_ful_levels = floor( log2((double) (2*num_vertices - 1)));
	long int ful_vertices = pow((int) 2,(int) num_ful_levels) - 1;
	get_mask<<<bpg1,tpb1>>>(d_u,d_v,d_mask,num_vertices,1,ful_vertices);
	
	cudaFree(d_u);
	cudaFree(d_v);
	free(h_eu);
	free(h_ev);

	uint64_t *d_hash_value = NULL;
	size_t size_hash = 2*num_vals*sizeof(uint64_t);
	cudaMalloc((void **)&d_hash_value, size_hash);
	get_hash<<<bpg,tpb>>>(d_mask,d_hash_value,num_vertices);

	size_t size_bits = num_bits * sizeof(bool);
	bool *d_bits = NULL;
        cudaMalloc((void **)&d_bits, size_bits);
	cudaMemcpy(d_bits, h_bits, size_bits, cudaMemcpyHostToDevice);

	size_t size_f = sizeof(int);
	int *d_found = NULL;
	int *h_found = (int *)malloc(size_f);
        cudaMalloc((void **)&d_found, size_f);
	h_found[0] = 0;
	cudaMemcpy(d_found,h_found,size,cudaMemcpyHostToDevice);
	check_bloom<<<num_vals,num_hashes>>>(d_found,d_bits,d_mask,d_hash_value,num_bits,num_hashes,num_vals);

	cudaFree(d_mask);
	cudaFree(d_hash_value);
	cudaFree(d_bits);

	cudaMemcpy(h_found,d_found,size,cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	
	if (*h_found>0) return 0;
	else return 1;

}*/

/*void Neighbours(int u, int num_vertices, int num_hashes, int num_bits, bool *h_bits, bool *h_neighs)
{

	uint64_t *d_hash_value = NULL;
	size_t size_hash = 2*num_vertices*sizeof(uint64_t);
	cudaMalloc((void **)&d_hash_value, size_hash);

	size_t size_bits = num_bits * sizeof(bool);
	bool *d_bits = NULL;
        cudaMalloc((void **)&d_bits, size_bits);
	cudaMemcpy(d_bits, h_bits, size_bits, cudaMemcpyHostToDevice);

	bool *d_neighs = NULL;
	size_t size_neighs = (num_vertices)*sizeof(bool);
	cudaMalloc((void **)&d_neighs, size_neighs);
	
	int num_ful_levels = floor( log2((double) (2*num_vertices - 1)));
	long int ful_vertices = pow((int) 2,(int) num_ful_levels) - 1;
	
	cudaMemset(d_neighs, 0, size_neighs);

	dim3 tpb2(32,32,1);
        dim3 bpg2(2,2,1);

	get_neighbours<<<bpg2,tpb2>>>(u, d_neighs, d_hash_value, d_bits, num_bits, num_hashes, num_vertices, ful_vertices);

	cudaMemcpy(h_neighs, d_neighs, size_neighs, cudaMemcpyDeviceToHost);

	cudaFree(d_neighs);
	cudaFree(d_hash_value);
	cudaFree(d_bits);
}*/

#define MAX 4039

int q[4039];
int front = 0;
int rear = -1;
int itemCount = 0;

int peek() {
   return q[front];
}

bool isEmpty() {
   return itemCount == 0;
}

bool isFull() {
   return itemCount == MAX;
}

int size() {
   return itemCount;
}  

void push(int data) {

   if(!isFull()) {
	
      if(rear == MAX-1) {
         rear = -1;            
      }       

      q[++rear] = data;
      itemCount++;
   }
}

int removeData() {
   int data = q[front++];
	
   if(front == MAX) {
      front = 0;
   }
	
   itemCount--;
   return data;  
}


int main ()
{

	const int INF = 1e8;
	int num_vertices, num_edges, num_hashes, num_bits;
	scanf("%d",&num_vertices);
	scanf("%d",&num_edges);
	scanf("%d",&num_bits);
	scanf("%d",&num_hashes);

	size_t size = num_edges * sizeof(int);
	//int num_vals = 2*num_vertices*(num_vertices-1);

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
	size_t size_hash = 2*num_vertices*sizeof(uint64_t);
	cudaMalloc((void **)&d_hash_value, size_hash);

	//size_t size_bits = num_bits * sizeof(bool);
	bool *d_bits = NULL;
        cudaMalloc((void **)&d_bits, size_bits);
	cudaMemcpy(d_bits, h_bits, size_bits, cudaMemcpyHostToDevice);

	bool *d_neighs = NULL;
	size_t size_neighs = (num_vertices)*sizeof(bool);
	cudaMalloc((void **)&d_neighs, size_neighs);
	
	int num_ful_levels = floor( log2((double) (2*num_vertices - 1)));
	long int ful_vertices = pow((int) 2,(int) num_ful_levels) - 1;
	
	cudaMemset(d_neighs, 0, size_neighs);

	dim3 tpb2(32,32,1);
        dim3 bpg2(2,2,1);

	int bfs_dist[4039];

	for (int i = 0; i < num_vertices; ++i) {
		bfs_dist[i] = INF;
	}

	push(0);
	bfs_dist[0] = 0;

	while(!isEmpty()){
		int u = removeData();
		//std::vector<int> adj;
		size_t size_neighs = (num_vertices)*sizeof(bool);
		bool *h_neighs = (bool *)malloc(size_neighs);
		//Neighbours(u, num_vertices, num_hashes, num_bits, h_bits, h_neighs);
		get_neighbours<<<bpg2,tpb2>>>(u, d_neighs, d_hash_value, d_bits, num_bits, num_hashes, num_vertices, ful_vertices);
		cudaMemcpy(h_neighs, d_neighs, size_neighs, cudaMemcpyDeviceToHost);

		for (int i = 0; i < num_vertices-1; ++i) {
			if (h_neighs[i] == 1){
				if (bfs_dist[i] == INF){
					bfs_dist[i] = bfs_dist[u] + 1;
					push(i);
				}
			}
		}
	}
	//if (DISPLAY_BFS_DIST) {
	//	for (int i = 0; i < num_vertices; ++i) {
	//		if (bfs_dist[i] != INF) {
	//			cout << i << " - " << bfs_dist[i] << "\n";
	//		}
	//	}
	//}

	//bool val = IsEdge(8, 7, num_vertices, num_hashes, num_bits, h_bits);
	//if (val == 0) printf("It is NOT an edge.\n");
	//else printf("It is an edge.\n");


	//size_t size_no = sizeof(int);
	//int *h_no = (int *)malloc(size_no);

	//size_t size_neighs = (num_vertices)*sizeof(bool);
	//bool *h_neighs = (bool *)malloc(size_neighs);

	//Neighbours(8, num_vertices, num_hashes, num_bits, h_bits, h_neighs);


	cudaFree(d_neighs);
	cudaFree(d_hash_value);
	cudaFree(d_bits);

	free(h_u);
	free(h_v);
	free(h_bits);
	
	cudaDeviceReset();
	return 0;
}
