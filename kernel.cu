#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h> 
#include "math.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "myrandom.h"
#include "periodicposition.h"
#define fcut 1E-8
#define pi 3.1415926535897932
//harmonical model,
__global__ void leapfrogone(double* Allcuda_position,double Dt,int n){
	//store the information of the particle in nine dimensional 1D array--(x,y,vx,vy,Fx,Fy,radius,Fx_normalAll,Fy_normalAll) and radius is the radius of the particle;
	//the second law of newton.
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < n){
		if (i == 0) printf("before leapfrog:x=%lf, vx=%lf y=%lf vy=%lf\n ", Allcuda_position[i * 9 + 0], Allcuda_position[i * 9 + 2], Allcuda_position[i * 9 + 1], Allcuda_position[i * 9 + 3]);
		Allcuda_position[i*9+0] = Allcuda_position[i*9+0] + Dt* Allcuda_position[i*9+2] + 1.0 /2.0 * Dt* Dt* Allcuda_position[i*9+4];
		if (i == 0) printf("After leapfrog:x=%lf, vx=%lf y=%lf vy=%lf\n", Allcuda_position[i * 9 + 0], Allcuda_position[i * 9 + 2], Allcuda_position[i * 9 + 1], Allcuda_position[i * 9 + 3]);
		Allcuda_position[i*9+1] = Allcuda_position[i*9+1] + Dt* Allcuda_position[i*9+3] + 1.0 / 2.0 * Dt* Dt* Allcuda_position[i*9+5];
		//the first step of leap-frog;
		Allcuda_position[i*9+2] =Allcuda_position[i*9+2] + Dt/2.0*Allcuda_position[i*9+4];
		Allcuda_position[i*9+3]=Allcuda_position[i*9+3] + Dt/2.0*Allcuda_position[i*9+5];
	}
}
__global__ void refreshforce(double* Allcuda_position,double scale,size_t n){
	int ith = blockDim.x*blockIdx.x + threadIdx.x;
	double deltax;
	double deltay;
	double rij;
	double sumforcex = 0;
	double sumforcey = 0;
	double dij;
	for (int j = 0; j < n; j++){
		if (ith == j) continue;
		deltax = (Allcuda_position[9 * ith + 0] - Allcuda_position[9 * j + 0]) / scale;
		deltay = (Allcuda_position[9 * ith + 1] - Allcuda_position[9 * j + 1]) / scale;
		deltax = (deltax - round(deltax))*scale;
		deltay = (deltay - round(deltay))*scale;
		rij = sqrt(deltax*deltax + deltay*deltay);//the distance between two particles.
		dij = (Allcuda_position[9 * j + 6] + Allcuda_position[9 * ith + 6]);
		if (rij> dij){
			sumforcex = sumforcex + 0.0;
			sumforcey = sumforcey + 0.0;
		}
		else
		{
			sumforcex = sumforcex + 2.0 / dij*(1 - rij / dij)*deltax / rij;
			sumforcey = sumforcey + 2.0 / dij*(1 - rij / dij)*deltay / rij;
		}

	}
	Allcuda_position[9 * ith + 4] = sumforcex;
	Allcuda_position[9 * ith + 5] = sumforcey;
}
__global__ void leapfrogtwo(double *Allcuda_position, double Dt, size_t n){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < n){
		//the second step of leapfrog
		Allcuda_position[i*9+2] = Allcuda_position[i*9+2] + Dt/ 2.0 * Allcuda_position[i*9+4];
		Allcuda_position[i*9+3] = Allcuda_position[i*9+3] + Dt/ 2.0 * Allcuda_position[i*9+5];
	}
}
__global__ void maxforce(double *Allcuda_position,double* max, size_t n){
	double maxforce=0;
	double force;
	for (int i = 0; i < n; i++){
		force = Allcuda_position[9 * i + 4] * Allcuda_position[9 * i + 4] + Allcuda_position[9 * i + 5] * Allcuda_position[9 * i + 5];
		if (maxforce < force) maxforce = force;
	}
	*max = maxforce;
}
//calcualte the keneticall energy for the system
__global__ void kenergy(double* Allcuda_position, double *energy,size_t n){
	energy[0] = 0;
	for (int i = 0; i < n; i++){
		energy[0] = energy[0]+Allcuda_position[9*i+2] * Allcuda_position[9*i+2];
		energy[0] = energy[0] + Allcuda_position[9*i+3]*Allcuda_position[9*i+3];
	}
	energy[0] = energy[0] / 2.0;
}
__global__ void penergy(double* Allcuda_position, double *energy,double scale, size_t n){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	energy[i] = 0;
	double deltax;
	double deltay;
	double rij;
	double dij;
		for (int j = 0; j < n; j++){
			if (j == i) continue;
			deltax = (Allcuda_position[9 * i + 0] - Allcuda_position[9 * j + 0]) / scale;
			deltay = (Allcuda_position[9 * i + 1] - Allcuda_position[9 * j + 1]) / scale;
			deltax = (deltax - round(deltax))*scale;
			deltay = (deltay - round(deltay))*scale;
			rij = sqrt(deltax*deltax + deltay*deltay);//the distance between two particles.
			dij = (Allcuda_position[9 * j + 6] + Allcuda_position[9 * i + 6]);
			if(rij>dij) energy[i]=energy[i]+0;
			else
			{
				energy[i] = energy[i] + (1 - rij / dij)*(1 - rij / dij);
			}
		}
	energy[i] = energy[i] / 2.0;
}
__host__ double power(double* Allhost_position, size_t n){
	//store the power of the particles and get the normalized force of the particles.
	double sumpower = 0;
	for (int i = 0; i < n; i++){
		sumpower = sumpower + Allhost_position[i * 9 + 2] * Allhost_position[i * 9 + 4];
		sumpower = sumpower + Allhost_position[i * 9 + 3] * Allhost_position[i * 9 + 5];
	}
	return sumpower;
}
__host__ double sumpenergy(double *host_penergy, double n){
	double sum = 0;
	for (int i = 0; i < n; i++){
		sum = sum + host_penergy[i];
	}
	return sum;
}
__host__ double sumkenergy(double *Allhost_position, size_t n){
	double sum=0;
	for (int i = 0; i < n; i++){
		sum = sum + Allhost_position[9 * i + 2] * Allhost_position[9 * i + 2];
		sum = sum + Allhost_position[9 * i + 3] * Allhost_position[9 * i + 3];
	}
	return sum / 2.0;
}
__global__ void forcenormal(double *Allcuda_postion, size_t n){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	double sum = 0;
	for (int j = 0; j < n; j++){
		sum = sum + Allcuda_postion[9 * j + 4] * Allcuda_postion[9 * j + 4];
		sum = sum + Allcuda_postion[9 * j + 5] + Allcuda_postion[9 * j + 5];
	}
	if (sum == 0){
		Allcuda_postion[9 + i + 7] = 0;
		Allcuda_postion[9 * i + 8] = 0;
	}
	else
	{
		Allcuda_postion[9 * i + 7] = Allcuda_postion[9 * i + 4] / sqrt(sum);
		Allcuda_postion[9 * i + 8] = Allcuda_postion[9 * i + 5] / sqrt(sum);
	}
}
__host__ double maxforce(double* Allhost_position, size_t n){
	double max = 0;
	double sum = 0; 
	for (int i = 0; i < n; i++){
		sum = 0;
		sum = Allhost_position[9 * i + 4] * Allhost_position[9 * i + 4] + sum;
		sum = Allhost_position[9 * i + 5] * Allhost_position[9 * i + 5] + sum;
		sum = sqrt(sum);
		if (max < sum) max = sum;
	}
	return max;
}
__global__ void setv(double* Allcuda_position,double alpha, size_t n){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	double sumv=0;
	for (int j = 0; j < n; j++){
		sumv = sumv + Allcuda_position[9 * j + 2] * Allcuda_position[9 * j + 2];
		sumv = sumv + Allcuda_position[9 * j + 3] * Allcuda_position[9 * j + 3];
	}
	Allcuda_position[i * 9 + 2] = (1 - alpha)*Allcuda_position[i * 9 + 2] + alpha*Allcuda_position[9 * i + 7]*sqrt(sumv);
	Allcuda_position[i * 9 + 3] = (1 - alpha)*Allcuda_position[i * 9 + 3] + alpha*Allcuda_position[9 * i + 8]*sqrt(sumv);
}
__global__ void freezev(double* Allcuda_position,size_t n){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	Allcuda_position[i * 9 + 2] = 0;
	Allcuda_position[i * 9 + 3] = 0;
}
int main(){
	/*init the position for all the particles*/
	double r_small = 0.5;
	double r_big = 1.4*0.5;
	double packing_fraction = 0.92;
	int N = 128*128;//216 big particles and 216 small particles
	//scale is the length of the square.
	double scale = sqrt(N / 2.0*pi*(r_small*r_small + r_big*r_big) / packing_fraction);
	double *Allhost_position = (double *)malloc(9 * N*sizeof(double));
	/*init all the parameters for each particles*/
	for (int i = 0; i < N / 2; i++) Allhost_position[9*i + 6] = r_small;//init the radius of small particles.
	for (int i = N / 2; i < N; i++) Allhost_position[9*i + 6] = r_big;//init the radius of big particles.
	for (int i = 0; i < N; i++){
		Allhost_position[9*i + 0] = genrand();//x
		Allhost_position[9*i + 0] = (Allhost_position[9*i + 0] - 0.5) * scale;//x
		Allhost_position[9*i + 1] = genrand();//y
		Allhost_position[9*i + 1] = (Allhost_position[9*i + 1] - 0.5)*scale;//y
		Allhost_position[9*i + 2] = 0;//vx
		Allhost_position[9*i + 3] = 0;//vy
	}
	/*end init place*/
	double* host_penergy = (double*)malloc(N*sizeof(double));
	double host_power;
	host_penergy[0] = 0;
	double* cuda_maxforce = NULL;
	double* cuda_penergy = NULL;
	double* Allcuda_position = NULL;
	cudaMalloc((void **)&cuda_maxforce, sizeof(double));
	cudaMalloc((void **)&cuda_penergy, N*sizeof(double));
	cudaMalloc((void **)&Allcuda_position, 9 * N*sizeof(double));
	cudaMemcpy(cuda_penergy, host_penergy, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Allcuda_position, Allhost_position, 9 * N*sizeof(double), cudaMemcpyHostToDevice);
	FILE *out, *out1;
	out = fopen("positionfirst.txt", "w+");
	out1 = fopen("positionsecond.txt", "w+");
	for (int i = 0; i < N; i++){
		fprintf(out, "%12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf\n", Allhost_position[9 * i + 0], Allhost_position[9 * i + 1], Allhost_position[9 * i + 6], Allhost_position[9 * i + 2], Allhost_position[9 * i + 3], Allhost_position[9 * i + 4], Allhost_position[9 * i + 5]);
	}
	int i = 0;
	refreshforce<<<16,1024>>>(Allcuda_position,scale,N);
	double max = 1;
	double f_inc = 1.1;
	double f_dec = 0.5;
	double alpha_start = 0.1;
	double alpha = 0.1;
	double f_alpha = 0.99;
	double Dt = 0.01;
	int Nmin=5;
	int count=0;
	while (max>fcut)
	{
		i++;
		if (i == 100000) break;
		std::cout << "This is the " << i << " step" << std::endl;
		leapfrogone<<<16,1024>>>(Allcuda_position,Dt,N);
		refreshforce<<<16,1024>>>(Allcuda_position, scale, N);
		leapfrogtwo<<<16, 1024>>>(Allcuda_position,Dt, N);
		penergy <<<16, 1024>> >(Allcuda_position, cuda_penergy, scale, N);
		cudaMemcpy(Allhost_position, Allcuda_position, 9 * N*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_penergy,cuda_penergy, N*sizeof(double), cudaMemcpyDeviceToHost);
		host_power=power(Allhost_position,N);
		std::cout << "the power is :"<<host_power << std::endl;
		std::cout << "the all energy is: " << sumpenergy(host_penergy, N) + sumkenergy(Allhost_position, N)<<std::endl;
		forcenormal << <16, 1024>> >(Allcuda_position, N);
		setv<<<16,1024>>>(Allcuda_position, alpha, N);
		max = maxforce(Allhost_position, N);
		std::cout << "the max force is: " << max << std::endl;
		if (host_power > 0&&count>5) {
			count = 0;
			Dt = Dt*f_inc < 10 * Dt ? Dt*f_inc : 10 * Dt;
			alpha = alpha*f_alpha;
		} 
		else if (host_power>0)
		{
			count = 0;
		}
		else
		{
			count++;
			Dt = Dt*f_dec;
			freezev<<<1,N>>>(Allcuda_position,N);
			alpha = alpha_start;
		}
	}
    periodic<<<16,1024>>>(Allcuda_position, scale, N);
    cudaMemcpy(Allhost_position, Allcuda_position, 9 * N*sizeof(double), cudaMemcpyDeviceToHost);
	for(int i = 0; i < N; i++){
		fprintf(out1, "%12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf %12.7lf\n", Allhost_position[9 * i + 0], Allhost_position[9 * i + 1], Allhost_position[9 * i + 6], Allhost_position[9 * i + 2], Allhost_position[9 * i + 3], Allhost_position[9 * i + 4], Allhost_position[9 * i + 5]);
	}
}
