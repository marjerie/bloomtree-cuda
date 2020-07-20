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

__device__ void MurmurHash3_x64_128 ( const char key[], const int len,
                           const uint32_t seed, void * out1 , void * out2)
{
  //const uint8_t * data = (const uint8_t*)key;

  uint8_t data[10];
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

  uint64_t blocks[10];
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

  uint8_t tail[10];
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
	//printf("%u and %u\n",hashA , hashB);
	return (hashA + n * hashB) % filter_size;
}

__global__ void init_mask(bool *mask, int n)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y)+blockIdx.y*gridDim.x+blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
	int tid = blockNum*(blockDim.x*blockDim.y*blockDim.z)+threadNum;
	if (tid < 2*n*(n-1))
		mask[tid] = 0;
}

__global__ void check_mask(bool *mask, int n)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y)+blockIdx.y*gridDim.x+blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
	int tid = blockNum*(blockDim.x*blockDim.y*blockDim.z)+threadNum;
	if (tid < 2*n*(n-1))
		mask[tid] = 0;
}

__global__ void adjust_level(int *u, int *v, bool *mask, int n, int e, long int ful_vertices)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid < e){
		u[tid] = u[tid] + n - 1;
		v[tid] = v[tid] + n - 1;
		__syncthreads();
		if (!((u[tid] < ful_vertices && v[tid] < ful_vertices) || (u[tid] >= ful_vertices && v[tid] >= ful_vertices))) {
			if (u[tid] > v[tid]){
				int src = u[tid];
				int dest = v[tid];
	 			int cur = Parent(u[tid]);
				mask[(cur*n+src-n+1) << 1] = 1;
				if (src == LeftChild(cur))
					mask[(cur*n+dest-n+1) << 1] = 1;
				else
					mask[((cur*n+dest-n+1) << 1) + 1] = 1;
				u[tid] = cur;
			}	
		 	else{
				int src = v[tid];
				int dest = u[tid];
	 			int cur = Parent(v[tid]);
				mask[(cur*n+src-n+1) << 1] = 1;
				if (src == LeftChild(cur))
					mask[(cur*n+dest-n+1) << 1] = 1;
				else
					mask[((cur*n+dest-n+1) << 1) + 1] = 1;
				v[tid] = cur;
			}
		}
	}
}

__global__ void calculate_lca(int *u, int *v, int *lca, int e)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	int val1 = 0;
	int val2 = 0;	

	if (tid < e){
		int i = 1;
		do{
			float pow_val = 1 << i;
			val1 = floor((u[tid]+1)/pow_val);
			val2 = floor((v[tid]+1)/pow_val);
			i++;
		} while(val1 != val2);
		lca[tid] = val1 - 1;
	}
}

__global__ void traversal(int *u, int *lca, int *source, int *destination, bool *mask, int n, int e)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;		

	if (tid < e){
		int prev = u[tid];
		int cur = Parent(u[tid]);
		int src = source[tid];
		int dest = destination[tid];
		while (cur != lca[tid]){
			mask[(cur*n+src) << 1] = 1;
			if (prev == LeftChild(cur))
				mask[(cur*n+dest) << 1] = 1;
			else
				mask[((cur*n+dest) << 1) + 1] = 1;
			prev = cur;
			cur = Parent(cur);
		}
		mask[((cur*n+src) << 1) + 1] = 1;
		mask[((cur*n+dest) << 1) + 1] = 1;
	}	
}

__global__ void to_str(char *str, bool *mask, int n, int count)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid<n){
		int num=tid;
		int val=0;
		int i =0;
		int len=0;
		do{
			len++;
			num/=10;
		}while(num !=0);
		num = tid;
		do{
			val=num%10 + 48;
			num/=10;
			char *op = (char *) (str+tid*sizeof(char)*(count+1)+len-i-1);
			*op = val;
			i++;
		}while(num !=0);
		char *op = (char *) (str+tid*sizeof(char)*(count+1)+len);
		*op = 48+'\0';
	}
}

__global__ void get_hash(bool *mask, uint64_t *hash_value, char *array, int n)
{
	int blockNum = blockIdx.z*(gridDim.x*gridDim.y)+blockIdx.y*gridDim.x+blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
	int tid = blockNum*(blockDim.x*blockDim.y*blockDim.z)+threadNum;

	if (tid < 2*n*(n-1)){
		if (mask[tid] == 1){
			int val=0;
			int i =0;
			int count=0; 	
			int num = tid;
			//uint64_t hash[2];
			do{	
				count++;
				num /= 10;
			} while(num != 0);
			num = tid;
			char str[10];
			do{
				val=num%10 + 48;
				num/=10;
				str[i] = val;
				i++;
			}while(num !=0);
			str[i] = 48+'\0';
			uint64_t len1 = (uint64_t) count;
			size_t len = (size_t) len1;
			MurmurHash3_x64_128(str, len, 0, (hash_value)+tid*2*sizeof(uint64_t), (hash_value)+(tid*2+1)*sizeof(uint64_t));
		}
	}
}

__global__ void set_bloom(bool *bit, bool *mask, uint64_t *hash_value, int m, int h, int n)
{
	int val = blockIdx.x;
	int hash = threadIdx.x;
	uint64_t filter_size = (uint64_t) m;
	uint8_t hash_no = (uint8_t) hash;
	
	if (val < n){
		if (mask[val] == 1){
			if (hash < h){
				bit[NthHash(hash_no,*(hash_value+2*val*sizeof(uint64_t)),*(hash_value+(2*val+1)*sizeof(uint64_t)), filter_size)] = 1;
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

int main ()
{
	cudaError_t err = cudaSuccess;

	int num_vertices, num_edges, num_hashes, num_bits;
	scanf("%d",&num_vertices);
	scanf("%d",&num_edges);
	scanf("%d",&num_bits);
	scanf("%d",&num_hashes);

	size_t size = num_edges * sizeof(int);
	int num_vals = 2*num_vertices*(num_vertices-1);

	int *h_u = (int *)malloc(size);
	int *h_v = (int *)malloc(size);

	if (h_u == NULL || h_u == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors u and v!\n");
            exit(EXIT_FAILURE);
        }
	
	for (int i =0; i<num_edges; i++)
	{
		scanf("%d",&h_u[i]);
		scanf("%d",&h_v[i]);
	}

	int *d_u = NULL, *d_v = NULL, *d_src = NULL, *d_dest = NULL;
        err= cudaMalloc((void **)&d_u, size);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector u (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err= cudaMalloc((void **)&d_v, size);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector v (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err= cudaMalloc((void **)&d_src, size);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector src (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err= cudaMalloc((void **)&d_dest, size);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector dest (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

	err= cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector u from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	err= cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector v from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	err= cudaMemcpy(d_src, h_u, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector src from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	err= cudaMemcpy(d_dest, h_v, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector dest from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

	size_t size_mask = num_vals * sizeof(bool);
	//bool *h_mask = (bool *)malloc(size_mask);
	bool *d_mask = NULL;
        err= cudaMalloc((void **)&d_mask, size_mask);

	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector mask (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

	dim3 tpb(num_vertices,(2*(num_vertices-1)),1);
	dim3 bpg(1,1,1);

	//dim3 tpb(32,32,1);
	//dim3 bpg(num_vertices/32,(num_vertices-1)/32,2);

	init_mask<<<bpg,tpb>>>(d_mask,num_vertices);

	err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch init_mask kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

	//int tpb1 = 1024;
        //int bpg1 = num_edges/1024;
	int tpb1 = num_edges;
        int bpg1 = 1;
	int num_ful_levels = floor( log2((double) (2*num_vertices - 1)));
	long int ful_vertices = pow((int) 2,(int) num_ful_levels) - 1;

	adjust_level<<<bpg1,tpb1>>>(d_u,d_v,d_mask,num_vertices,num_edges,ful_vertices);

	err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch adjust_level kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }	

	int *d_lca = NULL;
        err= cudaMalloc((void **)&d_lca, size);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector lca (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

	calculate_lca<<<bpg1,tpb1>>>(d_u,d_v,d_lca,num_edges);
	err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch calculate_lca kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	traversal<<<bpg1,tpb1>>>(d_u,d_lca,d_src,d_dest,d_mask,num_vertices,num_edges);
	err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch traversal kernel 1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	traversal<<<bpg1,tpb1>>>(d_v,d_lca,d_dest,d_src,d_mask,num_vertices,num_edges);
	err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch traversal kernel 2 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	
	err= cudaFree(d_lca);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector lca (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	err= cudaFree(d_u);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector lca (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	err= cudaFree(d_v);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector lca (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	err= cudaFree(d_src);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector lca (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	err= cudaFree(d_dest);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector lca (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }	

	uint64_t *d_hash_value = NULL;
	size_t size_hash = 2*num_vals*sizeof(uint64_t);
	err= cudaMalloc((void **)&d_hash_value, size_hash);
	//uint64_t *h_hash_value = (uint64_t *)malloc(size_hash);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector hash_value (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

	int num = num_vals;
	int count = 0;
	do{
		count++;
		num /= 10;
	} while(num != 0);
	
	char *d_str = NULL;
	size_t size_str = (count+1)*num_vals*sizeof(char);
	err= cudaMalloc((void **)&d_str, size_str);
	//char *h_str = (char *)malloc(size_str);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector str (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }	

	get_hash<<<bpg,tpb>>>(d_mask,d_hash_value,d_str,num_vertices);
	err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch get_hash kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

	size_t size_bits = num_bits * sizeof(bool);
	bool *h_bits = (bool *)malloc(size_bits);

	if (h_bits == NULL)
        {
            fprintf(stderr, "Failed to allocate host vector bits!\n");
            exit(EXIT_FAILURE);
        }

	bool *d_bits = NULL;
        err= cudaMalloc((void **)&d_bits, size_bits);
	if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector bits (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

	set_bloom<<<num_vals,num_hashes>>>(d_bits,d_mask,d_hash_value,num_bits,num_hashes,num_vals);
	err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch set_bloom kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
	
	dim3 tpb2(num_vertices,(4*(num_vertices-1)),1);
	dim3 bpg2(1,1,1);
	print_hash<<<bpg2,tpb2>>>(d_hash_value,num_vertices);
	
	cudaDeviceSynchronize();	
	cudaMemcpy(h_bits, d_bits, size_bits, cudaMemcpyDeviceToHost);

	cudaFree(d_mask);
	cudaFree(d_str);
	cudaFree(d_hash_value);
	cudaFree(d_bits);

	free(h_u);
	free(h_v);
	free(h_bits);
	
	cudaDeviceReset();
	return 0;
}