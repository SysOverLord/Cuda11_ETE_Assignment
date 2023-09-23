
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>

#include<windows.h>
//for linux
//#include<unistd.h>


struct Runner {
	int dist;
	int vel;
};

struct Team {
	Runner* runners;
	int curRunner;
	int id;
};


cudaError_t createTeamsWithCuda(Team* teams, Runner* runners, const int size);

cudaError_t simulateRaceWithCuda(Team* teams, Runner* runners, int* finished_team_count, int* placements, const int size);

__device__ int lock = 0;

__global__ void createTeamKernel(Team* teams, Runner* runners)
{
	int i = threadIdx.x;
	//RUN_LIMIT
	int size = 4;
	teams[i].runners = &runners[i * size];
	for (int j = 0; j < size; j++)
	{
		teams[i].runners[j].dist = j * 100;
		//At first everybody is at stop.
		teams[i].runners[j].vel = 0;
	}
	teams[i].id = i + 1;
	teams[i].curRunner = 0;

}



__global__ void simulateRaceKernel(Team* teams, Runner* runners, int* finished_team_count, int* placements, int rand_seed)
{
	int i = threadIdx.x;
	teams[i].runners = &runners[i * 4];
	int* curRunner = &teams[i].curRunner;

	if (*curRunner == 4)
	{
		//This team has ended the race.
		return;
	}

	if (teams[i].runners[*curRunner].dist < (*curRunner + 1) * 100)
	{
		teams[i].runners[*curRunner].dist += teams[i].runners[*curRunner].vel;
	}

	if (teams[i].runners[*curRunner].dist >= (*curRunner + 1) * 100)
	{
		teams[i].runners[*curRunner].vel = 0;
		*curRunner += 1;
	}

	if (*curRunner == 4)
	{

		//Race condition can not be paralel
		//Because threads can get asynchronous counts
		//Other solution in my mind was before calling kernel for every 400 threads
		//Count the curRunner == 4 flag to check if all 400 teams have finished the race
		//But that meant every second (Each Run) program would need to count from scratch
		//So This SingleThreaded madness must take place here.
		//Because Some teams finish very early and at most there was only 30-40 threads on lock.

		bool leaveLoop = false;
		while (!leaveLoop) {
			if (atomicExch(&lock, 1u) == 0u) {
				placements[*finished_team_count] = i;
				if (*finished_team_count == 0)
				{
					printf("\nFirst team to arrive finish line is Team %d\n", i + 1);
					printf("-----------------------------------------------\n");
					for (int j = 0; j < 4; j++)
					{
						printf("Team %d Runner %d VEL:%d DIST:%d\n", i + 1, j + 1, teams[i].runners[j].vel, teams[i].runners[j].dist);
					}
					printf("-----------------------------------------------\n");
				}
				*finished_team_count += 1;
				leaveLoop = true;
				atomicExch(&lock, 0u);
			}
		}

	}


	else
	{
		curandState_t state;
		curand_init(rand_seed, i, 0, &state);

		teams[i].runners[*curRunner].vel = curand_uniform(&state) * 5 + 1;
	}
}





int main(int argc, char* argv[])
{
	//400 olucak ama test için 5
	const int TEAM_SIZE = 400;

	//pointer for objects
	Team* teams = new Team[TEAM_SIZE];
	Runner* runners = new Runner[TEAM_SIZE * 4];
	//Pointer for placements
	int* placements = new int[TEAM_SIZE];


	cudaError_t cudaStatus = createTeamsWithCuda(teams, runners, TEAM_SIZE);

	//Pointers must be reassigned because 
	//Pointer values on the objects are for gpu memory (video-ram)
	//They are needed to be repointed to cpu memory (ram or virtual ram)
	for (int i = 0; i < TEAM_SIZE; i++) {
		teams[i].runners = &runners[i * 4];
	}


	int* consoleTeams;
	int consoleSize = 0;
	int finished_team_count = 0;

	if (argc <= 1)
	{
		consoleTeams = new int[TEAM_SIZE];
		printf("No arguments are passed while running the program.\nPlease state which teams will be shown on the console.\n");
		printf("All numbers must be seperated by space\n");
		do {
			scanf("%d", &consoleTeams[consoleSize++]);
		} while (getchar() != '\n' && consoleSize < TEAM_SIZE);
	}
	else
	{
		consoleTeams = new int[argc - 1];
		for (int i = 1; i < argc; i++)
		{
			sscanf(argv[i], "%d", &consoleTeams[consoleSize++]);
		}
	}
	for (int i = 0; i < consoleSize; i++)
	{
		if (consoleTeams[i] <= 0)
		{
			printf("Can't give an argument below or equal to 0 or NaN. Teams start at 1.");
			exit(-1);
		}
		else if (consoleTeams[i] > TEAM_SIZE)
		{
			printf("Can't select non existent team");
			exit(-2);
		}

	}

	while (finished_team_count < TEAM_SIZE)
	{
		simulateRaceWithCuda(teams, runners, &finished_team_count, placements, TEAM_SIZE);
		//Pointers must be reassigned because 
		//Pointer values on the objects are for gpu memory (video-ram)
		//They are needed to be repointed to cpu memory (ram or virtual ram)
		for (int i = 0; i < TEAM_SIZE; i++)
		{
			teams[i].runners = &runners[i * 4];
		}

		for (int i = 0; i < consoleSize; i++)
		{
			printf("-------------------------\n");
			int outTeam = consoleTeams[i];
			for (int j = 0; j < 4; j++)
			{
				printf("Team %d Runner %d VEL:%d DIST:%d\n", outTeam, j + 1, teams[outTeam - 1].runners[j].vel, teams[outTeam - 1].runners[j].dist);
			}
		}


		printf("-------------------------\n");
		printf("Finished Team Count:%d\n", finished_team_count);
		printf("|||||||||||||||||||||||||\n");

		//Sleep function in windows is in milliseconds
		//1000 olucak
		Sleep(1 * 1000);
		//For linux based
		// It is in seconds for linux.
		//sleep(1);
	}

	printf("Race has ended The Results are\n");;

	for (int i = 0; i < TEAM_SIZE; i++)
	{
		printf("%d PLACE: TEAM %d\n", i + 1, teams[placements[i]].id);
	}



	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	free(teams);
	free(runners);
	free(placements);
	free(consoleTeams);
	return 0;
}

cudaError_t simulateRaceWithCuda(Team* teams, Runner* runners, int* finished_team_count, int* placements, const int size)
{
	Team* dev_teams;
	Runner* dev_runners;
	int* dev_placements;
	int* dev_count;
	cudaError_t cudaStatus;

	srand(time(NULL));
	int rand_seed = rand() % 500 + 1000;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_placements, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_placements, placements, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}



	cudaStatus = cudaMalloc((void**)&dev_teams, size * sizeof(Team));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_teams, teams, size * sizeof(Team), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_runners, size * sizeof(Runner) * 4);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_runners, runners, size * sizeof(Runner) * 4, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_count, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_count, finished_team_count, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy on flag failed!\n");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	simulateRaceKernel << <1, size >> > (dev_teams, dev_runners, dev_count, dev_placements, rand_seed);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "simulateRaceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simulateRaceKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.

	cudaStatus = cudaMemcpy(teams, dev_teams, size * sizeof(Team), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy on Teams failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(runners, dev_runners, size * sizeof(Runner) * 4, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy on Runners failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(finished_team_count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy on count failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(placements, dev_placements, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy on placements failed!");
		goto Error;
	}


Error:
	cudaFree(dev_teams);
	cudaFree(dev_runners);
	cudaFree(dev_count);
	cudaFree(dev_placements);

	return cudaStatus;

}


cudaError_t createTeamsWithCuda(Team* teams, Runner* runners, const int size)
{
	Team* dev_teams;
	Runner* dev_runners;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_teams, size * sizeof(Team));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_teams, teams, size * sizeof(Team), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_runners, size * sizeof(Runner) * 4);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_runners, runners, size * sizeof(Runner) * 4, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// Launch a kernel on the GPU with one thread for each element.
	createTeamKernel << <1, size >> > (dev_teams, dev_runners);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "createTeamKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	//printf("%d", dev_teams[0].runners[0].dist);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching createTeamKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.

	cudaStatus = cudaMemcpy(teams, dev_teams, size * sizeof(Team), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy on Teams failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(runners, dev_runners, size * sizeof(Runner) * 4, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy on Runners failed!");
		goto Error;
	}




Error:
	cudaFree(dev_teams);
	cudaFree(dev_runners);

	return cudaStatus;

}
