/*

 * LRNR_ARRAY_CUDA.cu
 *
 *  Created on: 2013-11-21
 *      Author: Qinghan Meng
 */

/**************************************************************************************
 *COMMON MISTAKE MADE WHEN RUNNING THIS PROGRAM
 *a.X0 is not defined
 *
 *
 *
 *
 *
 *
 *
 */

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<cuda.h>
#include<time.h>
#include<R.h>
#include<Rinternals.h>

#define ATTR_NUM 11
#define INST_NUM 1000
#define MAXSTR 1024

/*
 * data file struct
 */

typedef struct file_struct {
	char name[MAXSTR];
	int nrow;
	int ncol;
} fileStruct;

fileStruct data_file;

float** matrix;

typedef struct device_data {
	float** dev_x;
	float* dev_y;
	int nind;
	int np;
	float** dev_x_h;
} deviceData;

deviceData dev_matrix;

typedef struct cross_validation {
	int fold;
	int** dev_train;
	int** dev_test;
	int train_num;
	int test_num;
	int* dev_train_num;
	int* dev_test_num;
	int** dev_test_h;
	int** dev_train_h;
	float accuracy_threshhold;
	int error_threshhold;
        float ll_threshhold;
} crossValidation;
crossValidation cv;

typedef struct tread_struct {
	dim3 dim_grid;
	dim3 dim_block;
} treadStruct;

treadStruct thread_config;

int maxThreadsPerBlock;
int maxBlocksPerGrid;

typedef struct features_combn_result {
	//int* feature1;
	//int* feature2;
	//int* feature3;
	//int* error_num;
	int size;
	int num;
	int unit_size;
	float* features_error;  //store features and error

} FeaturesCombnResult;
FeaturesCombnResult result;

int n_comb = 2;

/*
 *********************************************************
 fix some features, add more features into the set. 
 all features in the model should not be greater than
 ATTR_NUM(10). 
 *********************************************************
 */
int fixed_features_num = 0;
int fixed_features[10];

__device__ int fixed_features_num_dev;
__device__ int fixed_features_dev[10];

int* fixed_features_set;
int fixed_features_set_size;
int fixed_features_size;

float** read_csv(char* filename, int nrow, int ncol) {

	float** matrix = (float**) malloc(nrow * sizeof(float*));
	int i;
	for (i = 0; i < nrow; ++i) {
		matrix[i] = (float*) malloc(ncol * sizeof(float));
	}

	FILE *fp;
	char StrLine[50000];

	if ((fp = fopen(filename, "r")) == NULL) {
		return matrix;
		
	}

	int row = 0;

	while (!feof(fp)) {
		if (row >= nrow) {
			break;
		}
		fgets(StrLine, 50000, fp);
		if (strlen(StrLine) == 0) {
			break;
		}

		int j = 0;
		int k = j;
		int col = 0;
		char temp[1024];
		while (StrLine[j] != '\0') {
			if (StrLine[j] != ',' && StrLine[j] != '\n' && StrLine[j] != '\t') {
				temp[k] = StrLine[j];
			} else {
				temp[k] = '\0';
				float e = atof(temp);
				matrix[row][col] = e;
				col++;
				k = -1;
			}
			j++;
			k++;
		}
		row++;

	}
	fclose(fp);
	return matrix;
}

int checkError(cudaError_t err) {
	if (err != cudaSuccess) {
		//printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,__LINE__);
                return -1;
	} else {
		return 0;
	}
}

__device__ void initMatrixATTRATTR(float matrix[ATTR_NUM][ATTR_NUM]) {
	int i, j;
	for (i = 0; i < ATTR_NUM; ++i) {
		for (j = 0; j < ATTR_NUM; ++j) {
			matrix[i][j] = 0;
		}
	}
}

__device__ void initArrayINST(float arr[]) {
	int i;
	for (i = 0; i < INST_NUM; ++i) {
		arr[i] = 0;
	}
}

__device__ void initArrayATTR(float arr[]) {
	int i;
	for (i = 0; i < ATTR_NUM; ++i) {
		arr[i] = 0;
	}
}

__device__ void initMatrixATTRINST(float matrix[ATTR_NUM][INST_NUM]) {
	int i, j;
	for (i = 0; i < ATTR_NUM; ++i) {
		for (j = 0; j < INST_NUM; ++j) {
			matrix[i][j] = 0;
		}
	}
}

__device__ float SIGN(float a, float b) {
	return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

__device__ float MAX(float a, float b) {
	return b > a ? (b) : (a);
}

__device__ float MIN(float a, float b) {
	return b < a ? (b) : (a);
}

__device__ float SQR(float a) {
	return a * a;
}

__device__ float pythag(const float a, const float b) {
	float absa, absb;

	absa = fabs(a);
	absb = fabs(b);
	if (absa > absb)
		return absa * sqrt(1.0 + SQR(absb / absa));
	else
		return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}

__device__ int svdcmp(float a[ATTR_NUM][ATTR_NUM], float w[],
		float v[ATTR_NUM][ATTR_NUM], int np) {
	int flag;
	int i, its, j, jj, k, l, nm;
	float anorm, c, f, g, h, s, scale, x, y, z;
	float volatile temp;
	int m = np;

	int n = np;

	float rv1[ATTR_NUM];
	g = scale = anorm = 0.0;
	for (i = 0; i < n; i++) {
		l = i + 2;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m) {
			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale != 0.0) {
				for (k = i; k < m; k++) {
					a[k][i] /= scale;
					s += a[k][i] * a[k][i];
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][i] = f - g;
				for (j = l - 1; j < n; j++) {
					for (s = 0.0, k = i; k < m; k++)
						s += a[k][i] * a[k][j];
					f = s / h;
					for (k = i; k < m; k++)
						a[k][j] += f * a[k][i];
				}
				for (k = i; k < m; k++)
					a[k][i] *= scale;
			}
		}
		w[i] = scale * g;
		g = s = scale = 0.0;
		if (i + 1 <= m && i + 1 != n) {
			for (k = l - 1; k < n; k++)
				scale += fabs(a[i][k]);
			if (scale != 0.0) {
				for (k = l - 1; k < n; k++) {
					a[i][k] /= scale;
					s += a[i][k] * a[i][k];
				}
				f = a[i][l - 1];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][l - 1] = f - g;
				for (k = l - 1; k < n; k++)
					rv1[k] = a[i][k] / h;
				for (j = l - 1; j < m; j++) {
					for (s = 0.0, k = l - 1; k < n; k++)
						s += a[j][k] * a[i][k];
					for (k = l - 1; k < n; k++)
						a[j][k] += s * rv1[k];
				}
				for (k = l - 1; k < n; k++)
					a[i][k] *= scale;
			}
		}
		anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	//__syncthreads();
	for (i = n - 1; i >= 0; i--) {
		if (i < n - 1) {
			if (g != 0.0) {
				for (j = l; j < n; j++)
					v[j][i] = (a[i][j] / a[i][l]) / g;
				for (j = l; j < n; j++) {
					for (s = 0.0, k = l; k < n; k++)
						s += a[i][k] * v[k][j];
					for (k = l; k < n; k++)
						v[k][j] += s * v[k][i];
				}
			}
			for (j = l; j < n; j++)
				v[i][j] = v[j][i] = 0.0;
		}
		v[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}
	//__syncthreads();
	for (i = MIN(m, n) - 1; i >= 0; i--) {
		l = i + 1;
		g = w[i];
		for (j = l; j < n; j++)
			a[i][j] = 0.0;
		if (g != 0.0) {
			g = 1.0 / g;
			for (j = l; j < n; j++) {
				for (s = 0.0, k = l; k < m; k++)
					s += a[k][i] * a[k][j];
				f = (s / a[i][i]) * g;
				for (k = i; k < m; k++)
					a[k][j] += f * a[k][i];
			}
			for (j = i; j < m; j++)
				a[j][i] *= g;
		} else
			for (j = i; j < m; j++)
				a[j][i] = 0.0;
		++a[i][i];
	}
	//__syncthreads();
	for (k = n - 1; k >= 0; k--) {
		for (its = 0; its < 30; its++) {
			flag = 1;
			for (l = k; l >= 0; l--) {
				nm = l - 1;
				temp = fabs(rv1[l]) + anorm;
				if (temp == anorm) {
					flag = 0;
					break;
				}
				temp = fabs(w[nm]) + anorm;
				if (temp == anorm)
					break;
			}
			if (flag) {
				c = 0.0;
				s = 1.0;
				for (i = l; i < k + 1; i++) {
					f = s * rv1[i];
					rv1[i] = c * rv1[i];
					temp = fabs(f) + anorm;
					if (temp == anorm)
						break;
					g = w[i];
					h = pythag(f, g);
					w[i] = h;
					h = 1.0 / h;
					c = g * h;
					s = -f * h;
					for (j = 0; j < m; j++) {
						y = a[j][nm];
						z = a[j][i];
						a[j][nm] = y * c + z * s;
						a[j][i] = z * c - y * s;
					}
				}
			}
			z = w[k];
			if (l == k) {
				if (z < 0.0) {
					w[k] = -z;
					for (j = 0; j < n; j++)
						v[j][k] = -v[j][k];
				}
				break;
			}
			if (its == 29)
				return 0; // cannot converge: multi-collinearity?
			x = w[l];
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = pythag(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
			c = s = 1.0;
			for (j = l; j <= nm; j++) {
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = c * g;
				z = pythag(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y *= c;
				for (jj = 0; jj < n; jj++) {
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = x * c + z * s;
					v[jj][i] = z * c - x * s;
				}
				z = pythag(f, h);
				w[j] = z;
				if (z) {
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = c * g + s * y;
				x = c * y - s * g;
				for (jj = 0; jj < m; jj++) {
					y = a[jj][j];
					z = a[jj][i];
					a[jj][j] = y * c + z * s;
					a[jj][i] = z * c - y * s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
	return 1;
}

__device__ void svd_inverse(float u[ATTR_NUM][ATTR_NUM], int* flag, int np) {

	//const float eps = 1e-24;
	const float eps = 1e-8;  // because the float

	int n = np;

	float w[ATTR_NUM];
	initArrayATTR(w);

	float v[ATTR_NUM][ATTR_NUM];
	initMatrixATTRATTR(v);

	(*flag) = svdcmp(u, w, v, np);

	//__syncthreads();

	// Look for singular values
	float wmax = 0;
	for (int i = 0; i < n; i++)
		wmax = w[i] > wmax ? w[i] : wmax;

	//__syncthreads();
	float wmin = wmax * eps;
	for (int i = 0; i < n; i++) {
		w[i] = w[i] < wmin ? 0 : 1 / w[i];
	}
	//__syncthreads();
	// u w t(v)

	// row U * 1/w

	// results matrix
	float r[ATTR_NUM][ATTR_NUM];

	for (int i = 0; i < n; i++) {

		for (int j = 0; j < n; j++) {
			r[i][j] = 0.0;
			u[i][j] = u[i][j] * w[j];
		}
	}
	//__syncthreads();
	// [nxn].[t(v)]
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			for (int k = 0; k < n; k++)
				r[i][j] += u[i][k] * v[j][k];

	//__syncthreads();

	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			u[i][j] = r[i][j];
		}

	//__syncthreads();

}

void InitGridBlock(int maxBlocks, int maxThreads) {
	maxBlocksPerGrid = maxBlocks;
	maxThreadsPerBlock = maxThreads;
}

void InitThreadConfig(int grid_x, int grid_y, int grid_z, int block_x,
		int block_y, int block_z) {
	thread_config.dim_grid = dim3(grid_x, grid_y, grid_z);
	thread_config.dim_block = dim3(block_x, block_y, block_z);
}

__device__ float logloss(float pred, float label) {
    float epsilon = 0.00001;
    float prob = 1 / (1 + exp(-pred));
    if(1- prob < epsilon){
         prob = 1 - epsilon;
    }

    if(prob < epsilon){
         prob = epsilon;
    }
    
    return -(label*logf(prob) + (1-label)*log(1-prob));
}

__device__ void fitLMNWithCV(int* dev_combn, int valid_num, int np, float** X,
	float* Y, int all_valid, int** train, int training_num, int** test,
	int test_num, int fold, float* ll) {

//int tid = threadIdx.x + blockIdx.x * blockDim.x;
int tid = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x
		+ threadIdx.x;

int nind = training_num;

if (tid >= valid_num) {
	return;
}

int features[ATTR_NUM];
features[0] = 0;

/*
 * add fixed features into the model.
 */
//for(int i = 0; i < fixed_features_num_dev; ++ i){
//    features[i+1] = fixed_features_dev[i];
//}
for (int i = 0; i < np - 1; i++) {
	features[i + 1] = dev_combn[tid * (np - 1) + i];
}

/*
 * change np to np + fixed_features_num_dev.
 */
//np = fixed_features_num_dev + np;
__syncthreads();

float coef[ATTR_NUM];
initArrayATTR(coef);

//float S[ATTR_NUM][ATTR_NUM];
//initMatrixATTRATTR(S);
float p[INST_NUM];
initArrayINST(p);

///////////////////////////////////////

// Newton-Raphson to fit logistic model

int converge = 0;
int it = 0;   //  迭代的次数

float T[ATTR_NUM][ATTR_NUM];
//			float T_T[ATTR_NUM][ATTR_NUM];

float ncoef[ATTR_NUM];

while (!converge && it < 20) {

	for (int i = 0; i < nind; i++) { //nind 训练集的个数
		float t = 0;
		for (int j = 0; j < np; j++) {  //np 变量个数

			t += coef[j] * X[train[fold][i]][features[j]];

		}
		p[i] = 1 / (1 + expf(-t)); //p[i] = 1 / (1 + exp(-t))

	}

	initMatrixATTRATTR(T);

	for (int j = 0; j < np; j++)
		for (int k = j; k < np; k++) {
			float sum = 0;
			for (int i = 0; i < nind; i++) {
				sum += X[train[fold][i]][features[j]] * (p[i] * (1 - p[i]))
						* X[train[fold][i]][features[k]];

			}
			T[j][k] = T[k][j] = sum;
		}

	int flag = 1;

//				initMatrixATTRATTR(T_T);
	svd_inverse(T, &flag, np);

	if (!flag) {
		all_valid = 0;
		return;
	}
	__syncthreads();

	initArrayATTR(ncoef);

	// note implicit transpose of X
	for (int i = 0; i < np; i++)
		for (int j = 0; j < nind; j++)
			for (int k = 0; k < np; k++) {

				ncoef[i] += (T[i][k] * X[train[fold][j]][features[k]])
						* (Y[train[fold][j]] - p[j]);
			}

	// Update coefficients, and check for
	// convergence4
	float delta = 0;
	for (int j = 0; j < np; j++) {
		delta += fabs(ncoef[j]);
		coef[j] += ncoef[j];
	}

	if (delta < 1e-6)
		converge = 1;

	// Next iteration
	it++;
	__syncthreads();

}

//int correct = 0;
float loss = 0.0;
for (int i = 0; i < test_num; ++i) {
	float t = 0;
	for (int j = 0; j < np; ++j) {
		t += coef[j] * X[test[fold][i]][features[j]];
	}
	//if ((t > 0 && Y[test[fold][i]] == 1.0)
	//		|| (t < 0 && Y[test[fold][i]] == 0.0)) {
	//	correct++;
	//}
	loss += logloss(t, Y[test[fold][i]]);
}


ll[tid] += loss;

}

__global__ void LRNCV(int* dev_combn, int valid_num, int np, float** X,
	float* Y, int all_valid, int** train, int* training_num, int** test,
	int* test_num, int fold, float* ll) {
for (int i = 0; i < fold; i++) {
	fitLMNWithCV(dev_combn, valid_num, np, X, Y, all_valid, train,
			training_num[i], test, test_num[i], i, ll);
	__syncthreads();

}

}

void InitResult(int n) {
result.size = 10000;
result.num = 0;
//result.feature1 = (int*)malloc(sizeof(int)*result.size);
//result.feature2 = (int*)malloc(sizeof(int)*result.size);
//result.feature3 = (int*)malloc(sizeof(int)*result.size);
//result.error_num = (int*)malloc(sizeof(int)*result.size);
result.unit_size = n;
result.features_error = (float*) malloc(sizeof(float) * n * result.size);
}

int IsInFixedFeatures(int feature, int i_fixed_features_set) {
for (int i = i_fixed_features_set * fixed_features_size;
		i < (i_fixed_features_set + 1) * fixed_features_size; i++) {
	if (feature == fixed_features_set[i]) {
		return 1;
	}
}
return 0;
}

void SearchCombn(int n, long long start, long long stop) {

int features_num = dev_matrix.np - 1;
int nind = dev_matrix.nind;
float** X = dev_matrix.dev_x;
float* Y = dev_matrix.dev_y;
int all_valid = 1;
int** train = cv.dev_train;
int** test = cv.dev_test;
int* training_num = cv.dev_train_num;
int* test_num = cv.dev_test_num;
int fold = cv.fold;
int error_threshhold = cv.error_threshhold;

float ll_threshhold = cv.ll_threshhold;

int* dev_combn;
float* dev_ll;

/*
 * following two variable should be configured automatically based on GPU performance.
 */

int num_combn = maxThreadsPerBlock * maxBlocksPerGrid;
int *combn = (int*) malloc(sizeof(int) * n * num_combn);
int num = 0;
cudaMalloc((void**) &dev_combn, sizeof(int) * n * num_combn);

float *ll = (float*) malloc(sizeof(float) * num_combn);
for (int h = 0; h < num_combn; h++) {
	ll[h] = 0.0;
}

cudaMalloc((void**) &dev_ll, sizeof(float) * num_combn);
cudaMemcpy(dev_ll, ll, sizeof(float) * num_combn, cudaMemcpyHostToDevice);

//char output_file_name[MAXSTR] = "output.csv";

//FILE *out = fopen(output_file_name,"w");

InitResult(n + 1);

long long counter = 0;

if (1 == n) {
	for (int i = 1; i <= features_num; i++) {

		counter++;
		if (counter < start) {
			continue;
		}
		if (counter > stop) {
			break;
		}

		if (num < num_combn) {
			combn[num * n] = i;
			num++;
		} else {
			//printf("finish combn 1 features, start exe in GPU.\n");
			//printf("i = %d,num = %d\n", i, num);
			cudaMemcpy(dev_combn, combn, sizeof(int) * n * num_combn,
					cudaMemcpyHostToDevice);

			InitThreadConfig(maxBlocksPerGrid, 1, 1, maxThreadsPerBlock, 1, 1);
			LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(
					dev_combn, num, n + 1, X, Y, all_valid,
					train, training_num, test, test_num, fold, dev_ll);
			cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn,
					cudaMemcpyDeviceToHost);
			for (int index = 0; index < num; ++index) {
				if (ll[index] <= ll_threshhold) {
					//fprintf(out,"%d,%d\n", combn[index * n],acc[index]);

					result.num += 1;
					if (result.num >= result.size) {
						result.size += 10000;
						//result.feature1 = (int*)realloc(result.feature1, result.size*sizeof(int));
						//result.error_num = (int*)realloc(result.error_num, result.size*sizeof(int));
						result.features_error = (float*) realloc(
								result.features_error,
								result.size * sizeof(float) * (n + 1));
					}

					for (int i_feature = 0; i_feature < n; i_feature++) {
						result.features_error[(result.num - 1) * (n + 1)
								+ i_feature] = combn[n * index + i_feature];
					}
					result.features_error[(result.num - 1) * (n + 1) + n] =
							ll[index];

				}

			}
			//fflush(out);

			for (int h = 0; h < num_combn; h++) {
				ll[h] = 0.0;
			}
			cudaMemcpy(dev_ll, ll, sizeof(float) * num_combn,
					cudaMemcpyHostToDevice);

			num = 0;
			combn[num * n] = i;
			num++;

		}

	}
	if (num != 0) {
		//printf("num = %d\n", num);
		cudaMemcpy(dev_combn, combn, sizeof(int) * n * num_combn,
				cudaMemcpyHostToDevice);
		InitThreadConfig((num - 1) / maxThreadsPerBlock + 1, 1, 1,
				maxThreadsPerBlock, 1, 1);
		LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(
				dev_combn, num, n+1, X, Y, all_valid, train,
				training_num, test, test_num, fold, dev_ll);
		cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn,
				cudaMemcpyDeviceToHost);
		for (int index = 0; index < num; ++index) {
			if (ll[index] <= ll_threshhold) {
				result.num += 1;
				if (result.num >= result.size) {
					result.size += 10000;
					//result.feature1 = (int*)realloc(result.feature1, result.size*sizeof(int));
					//result.error_num = (int*)realloc(result.error_num, result.size*sizeof(int));
					result.features_error = (float*) realloc(
							result.features_error,
							result.size * sizeof(float) * (n + 1));
				}

				for (int i_feature = 0; i_feature < n; i_feature++) {
					result.features_error[(result.num - 1) * (n + 1) + i_feature] =
							combn[n * index + i_feature];
				}
				result.features_error[(result.num - 1) * (n + 1) + n] =
						ll[index];
			}
		}
		//fflush(out);
	}

} else if (2 == n) {
	for (int i = 1; i <= features_num; i++) {
		for (int j = i + 1; j <= features_num; j++) {

			counter++;
			if (counter < start) {
				continue;
			}
			if (counter > stop) {
				break;
			}

			if (num < num_combn) {
				combn[num * n] = i;
				combn[num * n + 1] = j;

				num++;
			} else {
				//printf("finish combn 2 features, start exe in GPU.\n");
				//printf("i = %d,j = %d, num = %d\n", i, j,num);
				cudaMemcpy(dev_combn, combn, sizeof(int) * n * num_combn,
						cudaMemcpyHostToDevice);

				InitThreadConfig(maxBlocksPerGrid, 1, 1, maxThreadsPerBlock, 1,
						1);
				LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(
						dev_combn, num, n + 1, X, Y,
						all_valid, train, training_num, test, test_num,
						fold, dev_ll);
				cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn,
						cudaMemcpyDeviceToHost);
				for (int index = 0; index < num; ++index) {
					if (ll[index] <= ll_threshhold) {
						//fprintf(out,"%d,%d,%d\n", combn[index * n],combn[index * n + 1],acc[index]);
						result.num += 1;
						if (result.num >= result.size) {
							result.size += 10000;
							//result.feature1 = (int*)realloc(result.feature1, result.size*sizeof(int));
							//result.error_num = (int*)realloc(result.error_num, result.size*sizeof(int));
							result.features_error = (float*) realloc(
									result.features_error,
									result.size * sizeof(float) * (n + 1));
						}

						for (int i_feature = 0; i_feature < n; i_feature++) {
							result.features_error[(result.num - 1) * (n + 1)
									+ i_feature] = combn[n * index + i_feature];
						}
						result.features_error[(result.num - 1) * (n + 1) + n] =
								ll[index];
					}

				}
				//fflush(out);

				for (int h = 0; h < num_combn; h++) {
					ll[h] = 0.0;
				}
				cudaMemcpy(dev_ll, ll, sizeof(float) * num_combn,
						cudaMemcpyHostToDevice);

				num = 0;
				combn[num * n] = i;
				combn[num * n + 1] = j;
				num++;

			}
		}

	}
	if (num != 0) {
		//printf("num = %d\n", num);
		cudaMemcpy(dev_combn, combn, sizeof(int) * n * num_combn,
				cudaMemcpyHostToDevice);
		InitThreadConfig((num - 1) / maxThreadsPerBlock + 1, 1, 1,
				maxThreadsPerBlock, 1, 1);
		LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(
				dev_combn, num, n+1, X, Y, all_valid, train,
				training_num, test, test_num, fold, dev_ll);
		cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn,
				cudaMemcpyDeviceToHost);
		for (int index = 0; index < num; ++index) {
			if (ll[index] <= ll_threshhold) {
				result.num += 1;
				if (result.num >= result.size) {
					result.size += 10000;
					//result.feature1 = (int*)realloc(result.feature1, result.size*sizeof(int));
					//result.error_num = (int*)realloc(result.error_num, result.size*sizeof(int));
					result.features_error = (float*) realloc(
							result.features_error,
							result.size * sizeof(float) * (n + 1));
				}

				for (int i_feature = 0; i_feature < n; i_feature++) {
					result.features_error[(result.num - 1) * (n + 1) + i_feature] =
							combn[n * index + i_feature];
				}
				result.features_error[(result.num - 1) * (n + 1) + n] =
						ll[index];
			}

		}
		//fflush(out);
	}
} else if (3 == n) {

	for (int i = 1; i <= features_num; i++) {
		for (int j = i + 1; j <= features_num; j++) {
			for (int k = j + 1; k <= features_num; k++) {

				counter++;
				if (counter < start) {
					continue;
				}
				if (counter > stop) {
					break;
				}

				if (num < num_combn) {
					combn[num * n] = i;
					combn[num * n + 1] = j;
					combn[num * n + 2] = k;
					num++;
				} else {
					//printf("finish combn 3 features, start exe in GPU.\n");
					//printf("i = %d,j = %d, k = %d, num = %d\n", i, j, k,
					//			num);
					cudaMemcpy(dev_combn, combn, sizeof(int) * n * num_combn,
							cudaMemcpyHostToDevice);

					InitThreadConfig(maxBlocksPerGrid, 1, 1, maxThreadsPerBlock,
							1, 1);
					LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(
							dev_combn, num, n+1, X, Y,
							all_valid, train, training_num, test, test_num,
							fold, dev_ll);
					cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn,
							cudaMemcpyDeviceToHost);
					for (int index = 0; index < num; ++index) {
						if (ll[index] <= ll_threshhold) {
							result.num += 1;
							if (result.num >= result.size) {
								result.size += 10000;
								//result.feature1 = (int*)realloc(result.feature1, result.size*sizeof(int));
								//result.error_num = (int*)realloc(result.error_num, result.size*sizeof(int));
								result.features_error = (float*) realloc(
										result.features_error,
										result.size * sizeof(float) * (n + 1));
							}

							for (int i_feature = 0; i_feature < n;
									i_feature++) {
								result.features_error[(result.num - 1) * (n + 1)
										+ i_feature] = combn[n * index
										+ i_feature];
							}
							result.features_error[(result.num - 1) * (n + 1) + n] =
									ll[index];
						}
					}
					//fflush(out);

					for (int h = 0; h < num_combn; h++) {
						ll[h] = nind;
					}
					cudaMemcpy(dev_ll, ll, sizeof(int) * num_combn,
							cudaMemcpyHostToDevice);

					num = 0;
					combn[num * n] = i;
					combn[num * n + 1] = j;
					combn[num * n + 2] = k;
					num++;

				}
			}
		}
	}
	if (num != 0) {
		//printf("num = %d\n", num);
		cudaMemcpy(dev_combn, combn, sizeof(int) * n * num_combn,
				cudaMemcpyHostToDevice);
		InitThreadConfig((num - 1) / maxThreadsPerBlock + 1, 1, 1,
				maxThreadsPerBlock, 1, 1);
		LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(
				dev_combn, num, n+1, X, Y, all_valid, train,
				training_num, test, test_num, fold, dev_ll);
		cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn,
				cudaMemcpyDeviceToHost);
		for (int index = 0; index < num; ++index) {
			if (ll[index] <= ll_threshhold) {
				result.num += 1;
				if (result.num >= result.size) {
					result.size += 10000;
					//result.feature1 = (int*)realloc(result.feature1, result.size*sizeof(int));
					//result.error_num = (int*)realloc(result.error_num, result.size*sizeof(int));
					result.features_error = (float*) realloc(
							result.features_error,
							result.size * sizeof(float) * (n + 1));
				}

				for (int i_feature = 0; i_feature < n; i_feature++) {
					result.features_error[(result.num - 1) * (n + 1) + i_feature] =
							combn[n * index + i_feature];
				}
				result.features_error[(result.num - 1) * (n + 1) + n] =
						ll[index];
			}
		}
		//fflush(out);

	}

} else {
	//printf("%d combination is not supported now.\n", n);
	return;
}
//fclose(out);

}

void InitFixedFeatures(SEXP fixed_features_sexp, SEXP nrow, SEXP ncol) {

fixed_features_size = *INTEGER(ncol);
//test
//printf("%d\n", fixed_features_size);

fixed_features_set_size = *INTEGER(nrow);
//test
//printf("%d\n", fixed_features_set_size);
//test
//printf("%d\n", length(fixed_features_sexp));

fixed_features_set = (int*) malloc(sizeof(int) * length(fixed_features_sexp));

int* fixed_p = INTEGER(fixed_features_sexp);

for (int i = 0; i < length(fixed_features_sexp); i++) {
	fixed_features_set[i] = fixed_p[i];
	//test
	//printf("%d ", fixed_features_set[i]);
}
//printf("\n");

}

void SearchCombnFix(int n) {

int features_num = dev_matrix.np - 1;   // exclude X0
int nind = dev_matrix.nind;
float** X = dev_matrix.dev_x;
float* Y = dev_matrix.dev_y;
int all_valid = 1;
int** train = cv.dev_train;
int** test = cv.dev_test;
int* training_num = cv.dev_train_num;
int* test_num = cv.dev_test_num;
int fold = cv.fold;
//int error_threshhold = cv.error_threshhold;

float ll_threshhold = cv.ll_threshhold;

int* dev_combn;
float* dev_ll;  // device log loss


/*
 * following two variable should be configured automatically based on GPU performance.
 */

int num_combn = maxThreadsPerBlock * maxBlocksPerGrid;
//int *combn = (int*)malloc(sizeof(int)*n*num_combn);

int feature_num_in_model = n + fixed_features_size; // exclude x0
int *combn = (int*) malloc(sizeof(int) * feature_num_in_model * num_combn);
int num = 0;
cudaMalloc((void**) &dev_combn, sizeof(int) * feature_num_in_model * num_combn);

float *ll = (float*) malloc(sizeof(float) * num_combn);
for (int h = 0; h < num_combn; h++) {
	ll[h] = 0.0;
}

cudaMalloc((void**) &dev_ll, sizeof(float) * num_combn);
cudaMemcpy(dev_ll, ll, sizeof(float) * num_combn, cudaMemcpyHostToDevice);

//char output_file_name[MAXSTR] = "output.csv";

//FILE *out = fopen(output_file_name,"w");

InitResult(feature_num_in_model + 1);  // +1 mean error_num column

for (int i_set = 0; i_set < fixed_features_set_size; i_set++) {
	//printf("fixed_features_set_size = %d\n", fixed_features_set_size);
	if (1 == n) {
		//test
		//printf("1 == n features_num = %d\n", features_num);
		for (int i = 1; i <= features_num; i++) {
			if (IsInFixedFeatures(i, i_set)) {
				//test
				//printf("continue i = %d\n", i);
				continue;
			}

			if (num < num_combn) {
				//test
				//printf("num < num_combn\n");
				for (int i_fixed_feature = 0;
						i_fixed_feature < fixed_features_size;
						++i_fixed_feature) {
					combn[num * feature_num_in_model + i_fixed_feature] =
							fixed_features_set[i_set * fixed_features_size
									+ i_fixed_feature];
					//test
					//printf("%d ", combn[num*feature_num_in_model + i_fixed_feature]);
				}
				combn[num * feature_num_in_model + fixed_features_size] = i;
				//printf("%d ", combn[num * feature_num_in_model + fixed_features_size]);
				//test
				//printf("\n");
				num++;
			} else {
				//printf("finish combn 1 features, start exe in GPU.\n");
				//printf("i = %d,num = %d\n", i, num);
				cudaMemcpy(dev_combn, combn,
						sizeof(int) * feature_num_in_model * num_combn,
						cudaMemcpyHostToDevice);

				InitThreadConfig(maxBlocksPerGrid, 1, 1, maxThreadsPerBlock, 1,
						1);
				LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(
						dev_combn, num, feature_num_in_model + 1, X, Y, all_valid,
						train, training_num, test, test_num, fold, dev_ll);
				cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn,
						cudaMemcpyDeviceToHost);
				for (int index = 0; index < num; ++index) {
					if (ll[index] <= ll_threshhold) {
						result.num += 1;
						if (result.num >= result.size) {
							result.size += 10000;
							//result.feature1 = (int*)realloc(result.feature1, result.size*sizeof(int));
							//result.error_num = (int*)realloc(result.error_num, result.size*sizeof(int));
							result.features_error = (float*) realloc(
									result.features_error,
									result.size * sizeof(float)
											* (feature_num_in_model + 1));
						}
						//result.feature1[result.num - 1] =  combn[index * feature_num_in_model + fixed_features_size];
						//result.error_num[result_num - 1] = acc[index];
						for (int i_feature = 0;
								i_feature < feature_num_in_model; i_feature++) {
							result.features_error[(result.num - 1)
									* (feature_num_in_model + 1) + i_feature] =
									combn[feature_num_in_model * index
											+ i_feature];
						}
						result.features_error[(result.num - 1)
								* (feature_num_in_model + 1)
								+ feature_num_in_model] = ll[index];

					}

				}
				//fflush(out);

				for (int h = 0; h < num_combn; h++) {
					ll[h] = 0.0;
				}
				cudaMemcpy(dev_ll, ll, sizeof(float) * num_combn,
						cudaMemcpyHostToDevice);

				num = 0;

				//combn[num * n] = i;
				for (int i_fixed_feature = 0;
						i_fixed_feature < fixed_features_size;
						++i_fixed_feature) {
					combn[num * feature_num_in_model + i_fixed_feature] =
							fixed_features_set[i_set * fixed_features_size
									+ i_fixed_feature];
				}
				combn[num * feature_num_in_model + fixed_features_size] = i;

				num++;

			}

		}

	} else if (2 == n) {
		for (int i = 1; i <= features_num; i++) {
			for (int j = i + 1; j <= features_num; j++) {

				if (IsInFixedFeatures(i, i_set)
						|| IsInFixedFeatures(j, i_set)) {
					continue;
				}

				if (num < num_combn) {
					for (int i_fixed_feature = 0;
							i_fixed_feature < fixed_features_size;
							++i_fixed_feature) {
						combn[num * feature_num_in_model + i_fixed_feature] =
								fixed_features_set[i_set * fixed_features_size
										+ i_fixed_feature];
					}
					combn[num * feature_num_in_model + fixed_features_size] = i;
					combn[num * feature_num_in_model + fixed_features_size + 1] =
							j;
					num++;

				} else {
					//printf("finish combn 2 features, start exe in GPU.\n");
					//printf("i = %d,j = %d, num = %d\n", i, j,num);
					cudaMemcpy(dev_combn, combn,
							sizeof(int) * fixed_features_size * num_combn,
							cudaMemcpyHostToDevice);

					InitThreadConfig(maxBlocksPerGrid, 1, 1, maxThreadsPerBlock,
							1, 1);
					LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(
							dev_combn, num, fixed_features_size + 1, X, Y,
							all_valid, train, training_num, test, test_num,
							fold, dev_ll);
					cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn,
							cudaMemcpyDeviceToHost);
					for (int index = 0; index < num; ++index) {
						if (ll[index] <= ll_threshhold) {
							result.num += 1;
							if (result.num >= result.size) {
								result.size += 10000;
								result.features_error = (float*) realloc(
										result.features_error,
										result.size * sizeof(float)
												* (feature_num_in_model + 1));
							}
							//result.feature1[result.num - 1] =  combn[index * feature_num_in_model + fixed_features_size];
							//result.error_num[result_num - 1] = acc[index];
							for (int i_feature = 0;
									i_feature < feature_num_in_model;
									i_feature++) {
								result.features_error[(result.num - 1)
										* (feature_num_in_model + 1) + i_feature] =
										combn[feature_num_in_model * index
												+ i_feature];
							}
							result.features_error[(result.num - 1)
									* (feature_num_in_model + 1)
									+ feature_num_in_model] = ll[index];

						}

					}
					//fflush(out);

					for (int h = 0; h < num_combn; h++) {
						ll[h] = 0.0;
					}
					cudaMemcpy(dev_ll, ll, sizeof(float) * num_combn,
							cudaMemcpyHostToDevice);

					num = 0;
					for (int i_fixed_feature = 0;
							i_fixed_feature < fixed_features_size;
							++i_fixed_feature) {
						combn[num * feature_num_in_model + i_fixed_feature] =
								fixed_features_set[i_set * fixed_features_size
										+ i_fixed_feature];
					}
					combn[num * feature_num_in_model + fixed_features_size] = i;
					combn[num * feature_num_in_model + fixed_features_size + 1] =
							j;

					num++;

				}
			}

		}

	} else if (3 == n) {

		for (int i = 1; i <= features_num; i++) {
			for (int j = i + 1; j <= features_num; j++) {
				for (int k = j + 1; k <= features_num; k++) {
					if (IsInFixedFeatures(i, i_set)
							|| IsInFixedFeatures(j, i_set)
							|| IsInFixedFeatures(k, i_set)) {
						continue;
					}

					if (num < num_combn) {
						for (int i_fixed_feature = 0;
								i_fixed_feature < fixed_features_size;
								++i_fixed_feature) {
							combn[num * feature_num_in_model + i_fixed_feature] =
									fixed_features_set[i_set
											* fixed_features_size
											+ i_fixed_feature];
						}
						combn[num * feature_num_in_model + fixed_features_size] =
								i;
						combn[num * feature_num_in_model + fixed_features_size
								+ 1] = j;
						combn[num * feature_num_in_model + fixed_features_size
								+ 2] = k;
						num++;
					} else {
						//printf("finish combn 3 features, start exe in GPU.\n");
						//printf("i = %d,j = %d, k = %d, num = %d\n", i, j, k,
						//			num);
						cudaMemcpy(dev_combn, combn,
								sizeof(int) * n * num_combn,
								cudaMemcpyHostToDevice);

						InitThreadConfig(maxBlocksPerGrid, 1, 1,
								maxThreadsPerBlock, 1, 1);
						LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(
								dev_combn, num, n+1, X, Y,
								all_valid, train, training_num, test, test_num,
								fold, dev_ll);
						cudaMemcpy(ll, dev_ll, sizeof(int) * num_combn,
								cudaMemcpyDeviceToHost);
						for (int index = 0; index < num; ++index) {
							if (ll[index] <= ll_threshhold) {
								//fprintf(out,"%d,%d,%d,%d\n", combn[index * n],
								//	combn[index * n + 1], combn[index * n + 2],
								//	acc[index]);

								result.num += 1;
								if (result.num >= result.size) {
									result.size += 10000;
									result.features_error =
											(float*) realloc(
													result.features_error,
													result.size * sizeof(float)
															* (feature_num_in_model
																	+ 1));
								}

								for (int i_feature = 0;
										i_feature < feature_num_in_model;
										i_feature++) {
									result.features_error[(result.num - 1)
											* (feature_num_in_model + 1)
											+ i_feature] =
											combn[feature_num_in_model * index
													+ i_feature];
								}
								result.features_error[(result.num - 1)
										* (feature_num_in_model + 1)
										+ feature_num_in_model] = ll[index];

							}
						}
						//fflush(out);

						for (int h = 0; h < num_combn; h++) {
							ll[h] = 0.0;
						}
						cudaMemcpy(dev_ll, ll, sizeof(int) * num_combn,
								cudaMemcpyHostToDevice);

						num = 0;
						for (int i_fixed_feature = 0;
								i_fixed_feature < fixed_features_size;
								++i_fixed_feature) {
							combn[num * feature_num_in_model + i_fixed_feature] =
									fixed_features_set[i_set
											* fixed_features_size
											+ i_fixed_feature];
						}
						combn[num * feature_num_in_model + fixed_features_size] =
								i;
						combn[num * feature_num_in_model + fixed_features_size
								+ 1] = j;
						combn[num * feature_num_in_model + fixed_features_size
								+ 2] = k;
						num++;
						num++;

					}
				}
			}
		}

	} else {
		//printf("%d combination is not supported now.\n", n);
		return;
	}
	//printf("i_set = %d\n", i_set);

}

if (num != 0 & 1 == n) {

	
	cudaMemcpy(dev_combn, combn, sizeof(int) * feature_num_in_model * num_combn,
			cudaMemcpyHostToDevice);
	InitThreadConfig((num - 1) / maxThreadsPerBlock + 1, 1, 1,
			maxThreadsPerBlock, 1, 1);
	
	LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(dev_combn,
			num, feature_num_in_model + 1, X, Y, all_valid, train, training_num, test,
			test_num, fold, dev_ll);
	//test
	
	cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn, cudaMemcpyDeviceToHost);
	for (int index = 0; index < num; ++index) {

		if (ll[index] <= ll_threshhold) {
			//fprintf(out,"%d,%d\n", combn[index * n],acc[index]);
			//printf("acc : %d\n", acc[index]);
			result.num += 1;
			if (result.num >= result.size) {
				result.size += 10000;
				result.features_error = (float*) realloc(result.features_error,
						result.size * sizeof(float) * (feature_num_in_model + 1));
			}

			for (int i_feature = 0; i_feature < feature_num_in_model;
					i_feature++) {
				result.features_error[(result.num - 1)
						* (feature_num_in_model + 1) + i_feature] =
						combn[feature_num_in_model * index + i_feature];
				//test
				//printf("%d ", result.features_error[(result.num - 1)*(feature_num_in_model + 1) + i_feature]);
			}

			result.features_error[(result.num - 1) * (feature_num_in_model + 1)
					+ feature_num_in_model] = ll[index];
			//test
			//printf("%d ", result.features_error[(result.num - 1)*(feature_num_in_model + 1) + feature_num_in_model]);
			//printf("\n");
		}
	}
	//fflush(out);
}
if (num != 0 && 2 == n) {
	//printf("num = %d\n", num);
	cudaMemcpy(dev_combn, combn, sizeof(int) * feature_num_in_model * num_combn,
			cudaMemcpyHostToDevice);
	InitThreadConfig((num - 1) / maxThreadsPerBlock + 1, 1, 1,
			maxThreadsPerBlock, 1, 1);
	LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(dev_combn,
			num, feature_num_in_model + 1, X, Y, all_valid, train, training_num, test,
			test_num, fold, dev_ll);
	cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn, cudaMemcpyDeviceToHost);
	for (int index = 0; index < num; ++index) {
		if (ll[index] <= ll_threshhold) {

			result.num += 1;

			if (result.num >= result.size) {
				result.size += 10000;
				result.features_error = (float*) realloc(result.features_error,
						result.size * sizeof(float) * (feature_num_in_model + 1));
			}
			//result.feature1[result.num - 1] =  combn[index * feature_num_in_model + fixed_features_size];
			//result.error_num[result_num - 1] = acc[index];
			for (int i_feature = 0; i_feature < feature_num_in_model;
					i_feature++) {
				result.features_error[(result.num - 1)
						* (feature_num_in_model + 1) + i_feature] =
						combn[feature_num_in_model * index + i_feature];
			}
			result.features_error[(result.num - 1) * (feature_num_in_model + 1)
					+ feature_num_in_model] = ll[index];

		}
		//fflush(out);
	}
}

if (num != 0 && 3 == n) {
	//printf("num = %d\n", num);
	cudaMemcpy(dev_combn, combn, sizeof(int) * feature_num_in_model * num_combn,
			cudaMemcpyHostToDevice);
	InitThreadConfig((num - 1) / maxThreadsPerBlock + 1, 1, 1,
			maxThreadsPerBlock, 1, 1);
	LRNCV<<<thread_config.dim_grid, thread_config.dim_block>>>(
			dev_combn, num, feature_num_in_model+1, X, Y, all_valid, train,
			training_num, test, test_num, fold, dev_ll);
	cudaMemcpy(ll, dev_ll, sizeof(float) * num_combn, cudaMemcpyDeviceToHost);
	for (int index = 0; index < num; ++index) {
		if (ll[index] <= ll_threshhold) {

			result.num += 1;
			if (result.num >= result.size) {
				result.size += 10000;
				result.features_error = (float*) realloc(result.features_error,
						result.size * sizeof(float) * (feature_num_in_model + 1));
			}

			for (int i_feature = 0; i_feature < feature_num_in_model;
					i_feature++) {
				result.features_error[(result.num - 1)
						* (feature_num_in_model + 1) + i_feature] =
						combn[feature_num_in_model * index + i_feature];
			}
			result.features_error[(result.num - 1) * (feature_num_in_model + 1)
					+ feature_num_in_model] = ll[index];
		}
	}
	//fflush(out);

}

}

float* predict(float** test, int nind, int np, float* coef) {
int i, j;
float* lable = (float*) malloc(nind * sizeof(float));

for (i = 0; i < nind; ++i) {
	float t = 0;
	for (j = 0; j < np; ++j) {
		t += coef[j] * test[i][j];
	}
	lable[i] = t;
}
return lable;
}

void InitCrossValidation(float* y, int nind, int fold) {

int lable_one_num = 0;
int lable_zero_num = 0;
int *train_num = (int*) malloc(sizeof(int) * fold);
int *test_num = (int*) malloc(sizeof(int) * fold);

int* lable_one = (int*) malloc(sizeof(int) * nind);
int* lable_zero = (int*) malloc(sizeof(int) * nind);

for (int i = 0; i < nind; i++) {
	if ((int) y[i] == 0) {
		lable_one[lable_one_num] = i;
		lable_one_num++;
	} else {
		lable_zero[lable_zero_num] = i;
		lable_zero_num++;
	}
}
int lable_one_block = lable_one_num / fold;
int lable_one_left = lable_one_num % fold;
int lable_zero_block = lable_zero_num / fold;
int lable_zero_left = lable_zero_num % fold;

int** train = (int**) malloc(sizeof(int*) * fold);
for (int i = 0; i < fold; i++) {
	train[i] = (int*) malloc(sizeof(int) * nind);
}

int** test = (int**) malloc(sizeof(int*) * fold);
for (int i = 0; i < fold; i++) {
	test[i] = (int*) malloc(sizeof(int) * nind);
}

for (int i = 0; i < fold; i++) {
	// test
	test_num[i] = 0;
	for (int j = lable_one_block * i; j < lable_one_block * (i + 1); j++) {
		test[i][test_num[i]] = lable_one[j];
		test_num[i]++;
	}
	if (i < lable_one_left) {
		test[i][test_num[i]] = lable_one[fold * lable_one_block + i];
		test_num[i]++;
	}

	for (int j = lable_zero_block * i; j < lable_zero_block * (i + 1); j++) {
		test[i][test_num[i]] = lable_zero[j];
		test_num[i]++;
	}
	if (i < lable_zero_left) {
		test[i][test_num[i]] = lable_zero[fold * lable_zero_block + i];
		test_num[i]++;
	}

	//train
	train_num[i] = 0;
	for (int j = 0; j < lable_one_num; j++) {
		if ((j >= lable_one_block * i & j < lable_one_block * (i + 1))
				|| j == fold * lable_one_block + i) {
			continue;
		} else {
			train[i][train_num[i]] = lable_one[j];
			train_num[i]++;
		}
	}
	for (int j = 0; j < lable_zero_num; j++) {
		if ((j >= lable_zero_block * i & j < lable_zero_block * (i + 1))
				|| j == fold * lable_zero_block + i) {
			continue;
		} else {
			train[i][train_num[i]] = lable_zero[j];
			train_num[i]++;
		}
	}

}

int** dev_test_h = (int**) malloc(sizeof(int*) * fold);
for (int i = 0; i < fold; i++) {
	cudaMalloc((void**) &dev_test_h[i], nind * sizeof(int));
	cudaMemcpy(dev_test_h[i], test[i], sizeof(int) * nind,
			cudaMemcpyHostToDevice);
}
int** dev_test;
cudaMalloc((void**) &dev_test, fold * sizeof(int*));
cudaMemcpy(dev_test, dev_test_h, sizeof(int*) * fold, cudaMemcpyHostToDevice);

int** dev_train_h = (int**) malloc(sizeof(int*) * fold);
for (int i = 0; i < fold; i++) {
	cudaMalloc((void**) &dev_train_h[i], nind * sizeof(int));
	cudaMemcpy(dev_train_h[i], train[i], sizeof(int) * nind,
			cudaMemcpyHostToDevice);
}
int** dev_train;
cudaMalloc((void**) &dev_train, fold * sizeof(int*));
cudaMemcpy(dev_train, dev_train_h, sizeof(int*) * fold, cudaMemcpyHostToDevice);

int* dev_train_num;
cudaMalloc((void**) &dev_train_num, sizeof(int) * fold);
cudaMemcpy(dev_train_num, train_num, sizeof(int) * fold,
		cudaMemcpyHostToDevice);

int* dev_test_num;
cudaMalloc((void**) &dev_test_num, sizeof(int) * fold);
cudaMemcpy(dev_test_num, test_num, sizeof(int) * fold, cudaMemcpyHostToDevice);

cv.dev_train = dev_train;
cv.dev_test = dev_test;
cv.dev_train_num = dev_train_num;
cv.dev_test_num = dev_test_num;
cv.fold = fold;
cv.dev_train_h = dev_train_h;
cv.dev_test_h = dev_test_h;

//	// test
//	for(int i = 0; i < fold; i ++){
//		printf("fold %d\n", i);
//		printf("train num %d\n", train_num[i]);
//		for(int j = 0; j < train_num[i]; j ++){
//			printf("%d,", train[i][j]);
//		}
//		printf("\n");
//		printf("test num %d\n", test_num[i]);
//		for(int j = 0; j < test_num[i]; j++) {
//			printf("%d,", test[i][j]);
//		}
//		printf("\n");
//
//	}

}

void InitDeviceData() {
int nind = data_file.nrow;
int np = data_file.ncol - 1;

int i, j, k;

/*
 * alloc Y memory
 */

float* Y = (float*) malloc(nind * sizeof(float));

/*
 * init Y memory.
 */

for (i = 0; i < nind; i++) {
	Y[i] = matrix[i][np];
}

/*
 * initialize cross validation
 */
InitCrossValidation(Y, nind, cv.fold);

/*
 * alloc X memory in device
 */
float** dev_x_h = (float**) malloc(sizeof(float*) * nind);
for (int i = 0; i < nind; ++i) {
	cudaMalloc((void**) &dev_x_h[i], np * sizeof(float));
	//printf("%d:%lx\n",i,dev_x_h[i]);
	cudaMemcpy(dev_x_h[i], matrix[i], sizeof(float) * np,
			cudaMemcpyHostToDevice);
}
float** dev_x;
cudaMalloc((void**) &dev_x, nind * sizeof(float*));
cudaMemcpy(dev_x, dev_x_h, sizeof(float*) * nind, cudaMemcpyHostToDevice);
/*
 * alloc Y memory in device.
 */

float* dev_y;
cudaMalloc((void**) &dev_y, nind * sizeof(float));
cudaMemcpy(dev_y, Y, sizeof(float) * nind, cudaMemcpyHostToDevice);

dev_matrix.dev_x = dev_x;
dev_matrix.dev_y = dev_y;
dev_matrix.nind = nind;
dev_matrix.np = np;
dev_matrix.dev_x_h = dev_x_h;

free(Y);

}

//x0 should be include in x.
extern "C" {
SEXP InitDeviceData(SEXP x, SEXP y) {
int nind = length(y);
int np = length(x) / nind;
double* rx = (double*) REAL(x);
double* ry = (double*) REAL(y);

/*
 * Store the data into a float array.
 */
float* rx_float = (float*) malloc(sizeof(float) * length(x));
for (int i = 0; i < length(x); i++) {
	rx_float[i] = rx[i];
}
float* ry_float = (float*) malloc(sizeof(float) * length(y));
for (int i = 0; i < length(y); i++) {
	ry_float[i] = ry[i];
}

InitCrossValidation(ry_float, nind, cv.fold);

/*
 * alloc X memory in device
 */

float** dev_x_h = (float**) malloc(sizeof(float*) * nind);
for (int i = 0; i < nind; ++i) {
	cudaMalloc((void**) &dev_x_h[i], np * sizeof(float));
	//printf("%d:%lx\n",i,dev_x_h[i]);
	cudaMemcpy(dev_x_h[i], rx_float + np * i, sizeof(float) * np,
			cudaMemcpyHostToDevice);
}
float** dev_x;
cudaMalloc((void**) &dev_x, nind * sizeof(float*));
cudaMemcpy(dev_x, dev_x_h, sizeof(float*) * nind, cudaMemcpyHostToDevice);

/*
 * alloc Y memory in device.
 */

float* dev_y;
cudaMalloc((void**) &dev_y, nind * sizeof(float));
cudaMemcpy(dev_y, ry_float, sizeof(float) * nind, cudaMemcpyHostToDevice);

dev_matrix.dev_x = dev_x;
dev_matrix.dev_y = dev_y;
dev_matrix.nind = nind;
dev_matrix.np = np;
dev_matrix.dev_x_h = dev_x_h;

/*
 * unit test x
 */
/*
 float* x_test = (float*)malloc(sizeof(float)*np);
 for(int i = 0; i <nind; ++i){
 cudaMemcpy(x_test, dev_x_h[i], np*sizeof(float), cudaMemcpyDeviceToHost);
 for(int j = 0; j < np ; ++ j){
 printf("%f ", x_test[j]);
 }
 printf("\n");
 }
 */

/*
 * unit test y
 */
/*
 float* y_test = (float*)malloc(sizeof(float)*nind);
 cudaMemcpy(y_test, dev_y, nind*sizeof(float), cudaMemcpyDeviceToHost);
 for(int i = 0; i < nind; ++ i){
 printf("%f ", y_test[i]);
 }
 printf("\n");
 */
free(rx_float);
free(ry_float);
return R_NilValue;
}
}

void FreeAll() {
int nind = data_file.nrow;

//	cudaFree(dev_coef);
//	free(lable);
//	free(coef);

for (int i = 0; i < cv.fold; i++) {
	cudaFree(cv.dev_test_h[i]);
	cudaFree(cv.dev_train_h[i]);
}
free(cv.dev_test_h);
free(cv.dev_train_h);
cudaFree(cv.dev_test);
cudaFree(cv.dev_train);

for (int i = 0; i < nind; i++) {
	free(matrix[i]);
	cudaFree(dev_matrix.dev_x_h[i]);
}
cudaFree(dev_matrix.dev_x);
free(dev_matrix.dev_x_h);
cudaFree(dev_matrix.dev_y);
free(matrix);

/*
 * free cross validation data
 */

}

int GetDeviceCount() {
int count;
cudaGetDeviceCount(&count);
return count;
}

//int main(int argc, char* argv[]){
//
//	/*
//	 * argv[1]:file name
//	 * argv[2]:row numbers of data matrix
//	 * argv[3]:column number of data matrix
//	 * argv[4]:1,2,3 way search
//	 * argv[5]:eroror theshhold
//	 * argv[6] fold
//	 * argv[7] device id
//	 * argv[8] task start index
//	 * argv[9] task stop index
//	 *
//	 * a command example
//	 * "/home/qinghan/workspace/cuda_lr/data/512_t_statistics.csv" 40 514 2 40 10 0 1 100
//	 */
//
//	float start_time, stop_time;
//	start_time = clock();
//
//	strcpy(data_file.name,argv[1]);
//    data_file.nrow = atoi(argv[2]);
//    data_file.ncol = atoi(argv[3]);
//    int n_comb = atoi(argv[4]);
//    cv.error_threshhold = atoi(argv[5]);
//    cv.fold = atoi(argv[6]);
//
//    int device_id = atoi(argv[7]);
//    long long start = atoll(argv[8]);
//    long long stop = atoll(argv[9]);
//
//    InitGridBlock(256,65535);
//    //InitFixedFeatures();
//    cudaSetDevice(device_id);
//
//    matrix = read_csv(data_file.name, data_file.nrow, data_file.ncol);
//
//    InitDeviceData();
//
//    SearchCombn(n_comb, start, stop);
//
//    //test result
//    for(int i = 0; i < result.num; i ++){
//    	printf("%d,%d\n", result.feature1[i], result.error_num[i]);
//    }
//
//    cudaError_t err;
//    err = cudaGetLastError();
//    printf("error = %s \n", cudaGetErrorString(err));
//
//	FreeAll();
//
//	stop = clock();
//	printf("Execution time is %f seconds\n", (stop_time - start_time) / (float)CLOCKS_PER_SEC);
//
//}

extern "C" {

SEXP TransferResultToR() {
SEXP result_sexp;

PROTECT(result_sexp = allocVector(REALSXP, result.num * result.unit_size));
double* result_sexp_p = REAL(result_sexp);

for (int i = 0; i < result.num * result.unit_size; i++) {
	result_sexp_p[i] = result.features_error[i];
}

UNPROTECT(1);
return (result_sexp);
}

}

extern "C" {

SEXP LRCUDA(SEXP x, SEXP y, SEXP num_comb, SEXP ll_threshhold, SEXP fold,
	SEXP device_id, SEXP start, SEXP stop) {

cudaSetDevice(*INTEGER(device_id));
cv.ll_threshhold = (*REAL(ll_threshhold));
//printf("%f\n", cv.error_threshhold);
cv.fold = *INTEGER(fold);
//printf("%d\n", cv.fold);
//printf("I am here\n");
InitDeviceData(x, y);

InitGridBlock(65535, 64);
//InitGridBlock(256,64);
//InitResult();
n_comb = *INTEGER(num_comb);
SearchCombn(n_comb, *INTEGER(start), *INTEGER(stop));

SEXP result = TransferResultToR();
return result;

}

}

extern "C" {
SEXP LRCUDAWithFixedVal(SEXP x, SEXP y, SEXP num_comb, SEXP ll_threshhold,
	SEXP fold, SEXP device_id, SEXP fixed_features, SEXP nrow, SEXP ncol) {

cudaSetDevice(*INTEGER(device_id));
cv.ll_threshhold = (*REAL(ll_threshhold));
//printf("%d\n", cv.error_threshhold);
cv.fold = *INTEGER(fold);
//printf("%d\n", cv.fold);
//printf("I am here\n");
InitDeviceData(x, y);
//printf("I am here too\n");
//initialize fixed feature set;

InitFixedFeatures(fixed_features, nrow, ncol);

InitGridBlock(32767, 64);
//InitResult();
n_comb = *INTEGER(num_comb);
//printf("following n_comb \n");

SearchCombnFix(n_comb);

SEXP result = TransferResultToR();
return result;

}

}

extern "C" {

SEXP test(SEXP A, SEXP B) {
int n = length(A);
SEXP C;
PROTECT(C = allocVector(REALSXP, n));
double* ra = REAL(A);
double* rb = REAL(B);
double* rc = REAL(C);

for (int i = 0; i < n; i++) {
	rc[i] = ra[i] + rb[i];
}

UNPROTECT(1);
return (C);
}

}

extern "C" {
SEXP getGPUCount() {
int count;
cudaGetDeviceCount(&count);
SEXP N;
PROTECT(N = allocVector(INTSXP, 1));
INTEGER(N)[0] = count;
UNPROTECT(1);
return (N);

}
}

