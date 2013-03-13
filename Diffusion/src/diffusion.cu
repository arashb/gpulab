/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: diffusion
* file:    diffusion.cu
*
* 
\******* PLEASE ENTER YOUR CORRECT STUDENT LOGIN, NAME AND ID BELOW *********/
const char* studentLogin = "p010";
const char* studentName  = "John Doe";
const int   studentID    = 1234567;
/****************************************************************************\
*
* In this file the following methods have to be edited or completed:
*
* diffuse_linear_isotrop_shared(const float  *d_input, ... )
* diffuse_linear_isotrop_shared(const float3 *d_input, ... )
* diffuse_nonlinear_isotrop_shared(const float  *d_input, ... )
* diffuse_nonlinear_isotrop_shared(const float3 *d_input, ... )
* compute_tv_diffusivity_shared
* compute_tv_diffusivity_joined_shared
* compute_tv_diffusivity_separate_shared
* jacobi_shared(float  *d_output, ... )
* jacobi_shared(float3 *d_output, ... )
* sor_shared(float  *d_output, ... )
* sor_shared(float3 *d_output, ... )
*
\****************************************************************************/


#define DIFF_BW 16
#define DIFF_BH 16

#define TV_EPSILON 0.1f


#include "diffusion.cuh"



const char* getStudentLogin() { return studentLogin; };
const char* getStudentName()  { return studentName; };
int         getStudentID()    { return studentID; };
bool checkStudentData() { return strcmp(studentLogin, "p010") != 0 && strcmp(studentName, "John Doe") != 0 && studentID != 1234567; };
bool checkStudentNameAndID() { return strcmp(studentName, "John Doe") != 0 && studentID != 1234567; };


//----------------------------------------------------------------------------
// Linear Diffusion
//----------------------------------------------------------------------------


// mode 0 gray: linear diffusion
__global__ void diffuse_linear_isotrop_shared(
  const float *d_input,
  float *d_output,
  float timeStep, 
  int nx, int ny,
  size_t pitch)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const int idx = y*pitch + x;
  //d_output[idx] = 0;
  __shared__ float u[DIFF_BW+2][DIFF_BH+2];

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = d_input[idx];

    if (x == 0)  u[0][ty] = u[tx][ty];
    else if (threadIdx.x == 0) u[0][ty] = d_input[idx-1];
    if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = d_input[idx+1];

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (threadIdx.y == 0) u[tx][0] = d_input[idx-pitch];
    if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = d_input[idx+pitch];
  }

  __syncthreads();

  // ### implement me ###
  if (x < nx && y < ny) {
	d_output[idx] = u[tx][ty] + timeStep * ( u[tx + 1][ty]
			+ u[tx - 1][ty] + u[tx][ty + 1]
			+ u[tx][ty - 1] - 4 * u[tx][ty]);
  }
}



// mode 0 interleaved: linear diffusion
__global__ void diffuse_linear_isotrop_shared
(
 const float3 *d_input,
 float3 *d_output,
 float timeStep,
 int nx, int ny,
 size_t pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  float3 imgValue;

  // load data into shared memory
  if (x < nx && y < ny) {

    imgValue = *( (float3*)imgP );
    u[tx][ty] = imgValue;

    if (x == 0)  u[0][ty] = imgValue;
    else if (threadIdx.x == 0) u[0][ty] = *( ((float3*)imgP)-1 );
    if (x == nx-1) u[tx+1][ty] = imgValue;
    else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *( ((float3*)imgP)+1 );

    if (y == 0)  u[tx][0] = imgValue;
    else if (threadIdx.y == 0) u[tx][0] = *( (float3*)(imgP-pitchBytes) );
    if (y == ny-1) u[tx][ty+1] = imgValue;
    else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
  }

  __syncthreads();

  float3 tmpValue;
  tmpValue.x = u[tx][ty].x + timeStep * (u[tx + 1][ty].x + u[tx - 1][ty].x
			+ u[tx][ty + 1].x + u[tx][ty - 1].x - 4 * u[tx][ty].x);
  
  tmpValue.y = u[tx][ty].y + timeStep * (u[tx + 1][ty].y + u[tx - 1][ty].y
			+ u[tx][ty + 1].y + u[tx][ty - 1].y - 4 * u[tx][ty].y);

  tmpValue.z = u[tx][ty].z + timeStep * (u[tx + 1][ty].z + u[tx - 1][ty].z
			+ u[tx][ty + 1].z + u[tx][ty - 1].z - 4 * u[tx][ty].z);
  if (x < nx && y < ny) 
  *((float3*)(((char*)d_output) + y*pitchBytes) + x) = tmpValue;
  
}




//----------------------------------------------------------------------------
// Non-linear Diffusion - explicit scheme
//----------------------------------------------------------------------------



// mode 1 gray: nonlinear diffusion
__global__ void diffuse_nonlinear_isotrop_shared
(
 const float *d_input,
 const float *d_diffusivity,
 float *d_output,
 float timeStep,
 int   nx,
 int   ny,
 size_t   pitch
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const int idx = y*pitch + x;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = d_input[idx];
    g[tx][ty] = d_diffusivity[idx];

    if (x == 0) {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = d_input[idx-1];
      g[0][ty] = d_diffusivity[idx-1];
    }
      
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = d_input[idx+1];
      g[tx+1][ty] = d_diffusivity[idx+1];
    }


    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (threadIdx.y == 0) {
      u[tx][0] = d_input[idx-pitch];
      g[tx][0] = d_diffusivity[idx-pitch];
    }
      
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    } 
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = d_input[idx+pitch];
      g[tx][ty+1] = d_diffusivity[idx+pitch];
    }
  }

  __syncthreads();

  
  // ### implement me ###


}



// mode 1 interleaved: nonlinear diffusion
__global__ void diffuse_nonlinear_isotrop_shared
(
 const float3 *d_input,
 const float3 *d_diffusivity,
 float3 *d_output,
 float timeStep,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* diffP = (char*)d_diffusivity + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float3 g[DIFF_BW+2][DIFF_BH+2];
  float3 value;


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = *( (float3*)imgP );
    g[tx][ty] = *( (float3*)diffP );

    if (x == 0) {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = *( ((float3*)imgP)-1 );
      g[0][ty] = *( ((float3*)diffP)-1 );
    }
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    } 
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = *( ((float3*)imgP)+1 );
      g[tx+1][ty] = *( ((float3*)diffP)+1 );
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    } 
    else if (threadIdx.y == 0) {
      u[tx][0] = *( (float3*)(imgP-pitchBytes) );
      g[tx][0] = *( (float3*)(diffP-pitchBytes) );
    }
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
      g[tx][ty+1] = *( (float3*)(diffP+pitchBytes) );
    }
  }

  __syncthreads();

  
  // ### implement me ###

}



// diffusivity computation for modes 1-3 gray
__global__ void compute_tv_diffusivity_shared
(
 const float *d_input,
 float *d_output,
 int   nx,
 int   ny,
 size_t   pitch
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const int idx = y*pitch + x;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = d_input[idx];

    if (x == 0)  u[0][ty] = u[tx][ty];
    else if (threadIdx.x == 0) u[0][ty] = d_input[idx-1];      
    if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = d_input[idx+1];

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (threadIdx.y == 0) u[tx][0] = d_input[idx-pitch];
    if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = d_input[idx+pitch];
  }

  __syncthreads();

 
  // make use of the constant TV_EPSILON

  // ### implement me ###
}


/*! Computes a joined diffusivity for an RGB Image:
 *  (g_R,g_G,g_B)(R,G,B) := 
 *  (g((R+G+B)/3),g((R+G+B)/3),g((R+G+B)/3))
 * */
__global__ void compute_tv_diffusivity_joined_shared
(
 const float3 *d_input,
 float3 *d_output,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = *( (float3*)imgP );

    if (x == 0)  u[0][ty] = u[tx][ty];
    else if (threadIdx.x == 0) u[0][ty] = *( ((float3*)imgP)-1 );
    if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *( ((float3*)imgP)+1 );

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (threadIdx.y == 0) u[tx][0] = *( (float3*)(imgP-pitchBytes) );
    if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
  }

  __syncthreads();
  
  
  // make use of the constant TV_EPSILON

  // ### implement me ###

}


/*! Computes a separate diffusivity for an RGB Image:
 *  (g_R,g_G,g_B)(R,G,B) := 
 *  (g(R),g(G),g(B))
 * */
__global__ void compute_tv_diffusivity_separate_shared
(
 const float3 *d_input,
 float3 *d_output,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = *( (float3*)imgP );

    if (x == 0)  u[threadIdx.x][ty] = u[tx][ty];
    else if (threadIdx.x == 0) u[0][ty] = *( ((float3*)imgP)-1 );
    if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *( ((float3*)imgP)+1 );

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (threadIdx.y == 0) u[tx][0] = *( (float3*)(imgP-pitchBytes) );
    if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
  }

  __syncthreads();

  
  // make use of the constant TV_EPSILON

  // ### implement me ###

}




//----------------------------------------------------------------------------
// Non-linear Diffusion - Jacobi scheme
//----------------------------------------------------------------------------



// mode 2 gray: Jacobi solver
__global__ void jacobi_shared
(
 float *d_output,
 const float *d_input,
 const float *d_original,
 const float *d_diffusivity,
 float weight,
 int   nx,
 int   ny,
 size_t   pitch
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int idx = y*pitch + x;

  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = d_input[idx];
    g[tx][ty] = d_diffusivity[idx];

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = d_input[idx-1];
      g[0][ty] = d_diffusivity[idx-1];
    }
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = d_input[idx+1];
      g[tx+1][ty] = d_diffusivity[idx+1];
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (threadIdx.y == 0) {
      u[tx][0] = d_input[idx-pitch];
      g[tx][0] = d_diffusivity[idx-pitch];
    }
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = d_input[idx+pitch];
      g[tx][ty+1] = d_diffusivity[idx+pitch];
    }
  }

  __syncthreads();

  
  // ### implement me ###

}



// mode 2 interleaved: Jacobi solver
__global__ void jacobi_shared
(
 float3 *d_output,
 const float3 *d_input,
 const float3 *d_original,
 const float3 *d_diffusivity,
 float weight,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* diffP = (char*)d_diffusivity + y*pitchBytes + x*sizeof(float3);

  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float3 g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = *( (float3*)imgP );
    g[tx][ty] = *( (float3*)diffP );

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = *( ((float3*)imgP)-1 );
      g[0][ty] = *( ((float3*)diffP)-1 );
    }
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = *( ((float3*)imgP)+1 );
      g[tx+1][ty] = *( ((float3*)diffP)+1 );
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (threadIdx.y == 0) {
      u[tx][0] = *( (float3*)(imgP-pitchBytes) );
      g[tx][0] = *( (float3*)(diffP-pitchBytes) );
    }
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
      g[tx][ty+1] = *( (float3*)(diffP+pitchBytes) );
    }
  }

  __syncthreads();

  
  // ### implement me ###


}



//----------------------------------------------------------------------------
// Non-linear Diffusion - Successive Over-Relaxation (SOR)
//----------------------------------------------------------------------------



// mode 3 gray: SOR solver
__global__ void sor_shared
(
 float *d_output,
 const float *d_input,
 const float *d_original,
 const float *d_diffusivity,
 float weight,
 float overrelaxation,
 int   nx,
 int   ny,
 size_t   pitch,
 int   red
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int idx = y*pitch + x;
  
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = d_input[idx];
    g[tx][ty] = d_diffusivity[idx];

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = d_input[idx-1];
      g[0][ty] = d_diffusivity[idx-1];
    }
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = d_input[idx+1];
      g[tx+1][ty] = d_diffusivity[idx+1];
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (threadIdx.y == 0) {
      u[tx][0] = d_input[idx-pitch];
      g[tx][0] = d_diffusivity[idx-pitch];
    }
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = d_input[idx+pitch];
      g[tx][ty+1] = d_diffusivity[idx+pitch];
    }
  }

  __syncthreads();


  // ### implement me ###

}



// mode 3 interleaved: SOR solver
__global__ void sor_shared
(
 float3 *d_output,
 const float3 *d_input,
 const float3 *d_original,
 const float3 *d_diffusivity,
 float weight,
 float overrelaxation,
 int   nx,
 int   ny,
 size_t   pitchBytes,
 int   red
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* diffP = (char*)d_diffusivity + y*pitchBytes + x*sizeof(float3);

  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float3 g[DIFF_BW+2][DIFF_BH+2];



  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = *( (float3*)imgP );
    g[tx][ty] = *( (float3*)diffP );

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (threadIdx.x == 0) {
      u[0][ty] = *( ((float3*)imgP)-1 );
      g[0][ty] = *( ((float3*)diffP)-1 );
    }
    if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else if (threadIdx.x == blockDim.x-1) {
      u[tx+1][ty] = *( ((float3*)imgP)+1 );
      g[tx+1][ty] = *( ((float3*)diffP)+1 );
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (threadIdx.y == 0) {
      u[tx][0] = *( (float3*)(imgP-pitchBytes) );
      g[tx][0] = *( (float3*)(diffP-pitchBytes) );
    }
    if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else if (threadIdx.y == blockDim.y-1) {
      u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
      g[tx][ty+1] = *( (float3*)(diffP+pitchBytes) );
    }
  }

  __syncthreads();

  
  // ### implement me ###


}




//----------------------------------------------------------------------------
// Host function
//----------------------------------------------------------------------------



void gpu_diffusion
(
 const float *input,
 float *output,
 int nx, int ny, int nc, 
 float timeStep,
 int iterations,
 float weight,
 int lagged_iterations,
 float overrelaxation,
 int mode,
 bool jointDiffusivity
 )
{
  int i,j;
  size_t pitchF1, pitchBytesF1, pitchBytesF3;
  float *d_input = 0;
  float *d_output = 0;
  float *d_diffusivity = 0;
  float *d_original = 0;
  float *temp = 0;

  dim3 dimGrid((int)ceil((float)nx/DIFF_BW), (int)ceil((float)ny/DIFF_BH));
  dim3 dimBlock(DIFF_BW,DIFF_BH);

  // Allocation of GPU Memory
  if (nc == 1) {

    cutilSafeCall( cudaMallocPitch( (void**)&(d_input), &pitchBytesF1, nx*sizeof(float), ny ) );
    cutilSafeCall( cudaMallocPitch( (void**)&(d_output), &pitchBytesF1, nx*sizeof(float), ny ) );
    if (mode) cutilSafeCall( cudaMallocPitch( (void**)&(d_diffusivity), &pitchBytesF1, nx*sizeof(float), ny ) );
    if (mode >= 2) cutilSafeCall( cudaMallocPitch( (void**)&(d_original), &pitchBytesF1, nx*sizeof(float), ny ) );

    cutilSafeCall( cudaMemcpy2D(d_input, pitchBytesF1, input, nx*sizeof(float), nx*sizeof(float), ny, cudaMemcpyHostToDevice) );
    if (mode >= 2) cutilSafeCall( cudaMemcpy2D(d_original, pitchBytesF1, d_input, pitchBytesF1, nx*sizeof(float), ny, cudaMemcpyDeviceToDevice) );

    pitchF1 = pitchBytesF1/sizeof(float);

  } else if (nc == 3) {

    cutilSafeCall( cudaMallocPitch( (void**)&(d_input), &pitchBytesF3, nx*sizeof(float3), ny ) );
    cutilSafeCall( cudaMallocPitch( (void**)&(d_output), &pitchBytesF3, nx*sizeof(float3), ny ) );
    if (mode) cutilSafeCall( cudaMallocPitch( (void**)&(d_diffusivity), &pitchBytesF3, nx*sizeof(float3), ny ) );
    if (mode >= 2) cutilSafeCall( cudaMallocPitch( (void**)&(d_original), &pitchBytesF3, nx*sizeof(float3), ny ) );

    cutilSafeCall( cudaMemcpy2D(d_input, pitchBytesF3, input, nx*sizeof(float3), nx*sizeof(float3), ny, cudaMemcpyHostToDevice) );
    if (mode >= 2) cutilSafeCall( cudaMemcpy2D(d_original, pitchBytesF3, d_input, pitchBytesF3, nx*sizeof(float3), ny, cudaMemcpyDeviceToDevice) );

  }


  // Execution of the Diffusion Kernel

  if (mode == 0) {   // linear isotropic diffision
    if (nc == 1) {
      for (i=0;i<iterations;i++) {
        diffuse_linear_isotrop_shared<<<dimGrid,dimBlock>>>(d_input, d_output, timeStep, nx, ny, pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
    else if (nc == 3) {
      for (i=0;i<iterations;i++) {
        diffuse_linear_isotrop_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_output,timeStep,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
  }
  else if (mode == 1) {  // nonlinear isotropic diffusion
    if (nc == 1) {

      for (i=0;i<iterations;i++) {
        compute_tv_diffusivity_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        diffuse_nonlinear_isotrop_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,d_output,timeStep,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
    else if (nc == 3) {
      for (i=0;i<iterations;i++) {
        if (jointDiffusivity)
          compute_tv_diffusivity_joined_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);
        else
          compute_tv_diffusivity_separate_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);


        cutilSafeCall( cudaThreadSynchronize() );

        diffuse_nonlinear_isotrop_shared<<<dimGrid,dimBlock>>>
          ((float3*)d_input,(float3*)d_diffusivity,(float3*)d_output,timeStep,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
  }
  else if (mode == 2) {    // Jacobi-method
    if (nc == 1) {
      for (i=0;i<iterations;i++) {
        compute_tv_diffusivity_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        for (j=0;j<lagged_iterations;j++) {
          jacobi_shared<<<dimGrid,dimBlock>>> (d_output,d_input,d_original,
            d_diffusivity,weight,nx,ny,pitchF1);

          cutilSafeCall( cudaThreadSynchronize() );

          temp = d_input;
          d_input = d_output;
          d_output = temp;
        }
      }
    }
    else if (nc == 3) {
      for (i=0;i<iterations;i++) {
        if (jointDiffusivity)
          compute_tv_diffusivity_joined_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);
        else
          compute_tv_diffusivity_separate_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        for (j=0;j<lagged_iterations;j++) {
          jacobi_shared<<<dimGrid,dimBlock>>>
            ((float3*)d_output,(float3*)d_input,
            (float3*)d_original,(float3*)d_diffusivity,
            weight,nx,ny,pitchBytesF3);

          cutilSafeCall( cudaThreadSynchronize() );

          temp = d_input;
          d_input = d_output;
          d_output = temp;
        }
      }
    }    
  }
  else if (mode == 3) {    // Successive Over Relaxation (Gauss-Seidel with extrapolation)
    if (nc == 1) {
      for (i=0;i<iterations;i++) {
        compute_tv_diffusivity_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        for(j=0;j<lagged_iterations;j++) {					
          sor_shared<<<dimGrid,dimBlock>>>(d_input,d_input,d_original,
            d_diffusivity,weight,overrelaxation,nx,ny,pitchF1, 0);

          cutilSafeCall( cudaThreadSynchronize() );

          sor_shared<<<dimGrid,dimBlock>>>(d_input,d_input,d_original,
            d_diffusivity,weight,overrelaxation,nx,ny,pitchF1, 1);

          cutilSafeCall( cudaThreadSynchronize() );
        }
      }
    }
    if (nc == 3) {
      for (i=0;i<iterations;i++) {
        if (jointDiffusivity)
          compute_tv_diffusivity_joined_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);
        else
          compute_tv_diffusivity_separate_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        for (j=0;j<lagged_iterations;j++) {
          sor_shared<<<dimGrid,dimBlock>>>
            ((float3*)d_input,(float3*)d_input,
            (float3*)d_original,(float3*)d_diffusivity,
            weight,overrelaxation,nx,ny,pitchBytesF3, 0);

          cutilSafeCall( cudaThreadSynchronize() );

          sor_shared<<<dimGrid,dimBlock>>>
            ((float3*)d_input,(float3*)d_input,
            (float3*)d_original,(float3*)d_diffusivity,
            weight,overrelaxation,nx,ny,pitchBytesF3, 1);

          cutilSafeCall( cudaThreadSynchronize() );
        }
      }
    }
  }


  if (nc == 1) {
    if (mode == 3) cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float), d_input, pitchBytesF1, nx*sizeof(float), ny, cudaMemcpyDeviceToHost) );
    else cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float), d_output, pitchBytesF1, nx*sizeof(float), ny, cudaMemcpyDeviceToHost) );
  } else if (nc == 3) {
    if (mode == 3) cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float3), d_input, pitchBytesF3, nx*sizeof(float3), ny, cudaMemcpyDeviceToHost) );
    else cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float3), d_output, pitchBytesF3, nx*sizeof(float3), ny, cudaMemcpyDeviceToHost) );
  }


  // clean up
  if (d_original) cutilSafeCall( cudaFree(d_original) );
  if (d_diffusivity) cutilSafeCall( cudaFree(d_diffusivity) );
  if (d_output) cutilSafeCall( cudaFree(d_output) );
  if (d_input)  cutilSafeCall( cudaFree(d_input) );
}