#ifndef PERIODICPOSITION_H
__global__ void periodic(double* Allcuda_position, double scale, double n){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < n){
		Allcuda_position[i * 9 + 0] = Allcuda_position[i * 9 + 0] / scale;
		Allcuda_position[i * 9 + 0] = (Allcuda_position[i * 9 + 0] - round(Allcuda_position[i * 9 + 0]))*scale;
		Allcuda_position[i * 9 + 1] = Allcuda_position[i * 9 + 1] / scale;
		Allcuda_position[i * 9 + 1] = (Allcuda_position[i * 9 + 1] - round(Allcuda_position[i * 9 + 1]))*scale;

	}
}
#endif