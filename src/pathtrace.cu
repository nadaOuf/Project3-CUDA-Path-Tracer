#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define BLOCKSIZE 8

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;

//Device variables
static Camera *dev_camera = NULL;
static Geom *dev_scene_geom = NULL;
static Material *dev_scene_material = NULL;
static RayState *dev_ray_array = NULL;


void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    
	cudaMalloc(&dev_camera, sizeof(Camera));
	cudaMemcpy(dev_camera, &cam, sizeof(Camera), cudaMemcpyHostToDevice);
	checkCUDAError("Problem with camera memcpy");

	cudaMalloc(&dev_scene_geom, hst_scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_scene_geom, hst_scene->geoms.data(), hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	checkCUDAError("Problem with scene geometry memcpy");

	cudaMalloc(&dev_scene_material, hst_scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_scene_material, hst_scene->materials.data(), hst_scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
	checkCUDAError("Problem with scene material memcpy");

	cudaMalloc(&dev_ray_array, pixelcount * sizeof(RayState));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
   
   cudaFree(dev_camera);
   cudaFree(dev_scene_geom);
   cudaFree(dev_scene_material);
   cudaFree(dev_ray_array);

    checkCUDAError("pathtraceFree");
}

/**
 * Example function to generate static and test the CUDA-GL interop.
 * Delete this once you're done looking at it!
 */
__global__ void generateNoiseDeleteMe(Camera cam, int iter, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // CHECKITOUT: Note that on every iteration, noise gets added onto
        // the image (not replaced). As a result, the image smooths out over
        // time, since the output image is the contents of this array divided
        // by the number of iterations.
        //
        // Your renderer will do the same thing, and, over time, it will become
        // smoother.
        image[index] += glm::vec3(u01(rng));
    }
}

/**
* Generate gittered ray within pixel from camera to the scene
*/
__global__ void generateFirstLevelRays(Camera* cam, RayState* rays) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	 if (x < cam->resolution.x && y < cam->resolution.y) {
		int index = x + (y * cam->resolution.x);
	
		//Set the ray start position which is the camera position
		rays[index].ray.origin = cam->position;

		//Set the ray direction
		// 1. Get the pixel position in the world coordinates
		glm::vec3 pixelWorld = cam->mPosition + (cam->hVector * ((2.0f * (x*1.0f/(cam->resolution.x - 1.0f)) ) - 1) ) + (cam->vVector * (1 - (2.0f * (y*1.0f/(cam->resolution.y - 1.0f))) ) );
		
		// 2. Get the normalized ray direction from the ray origin (camera position) to the pixel world coordinates
		rays[index].ray.direction = glm::normalize(pixelWorld - cam->position);

		//Set the color carried by the ray to white
		rays[index].color = glm::vec3(1.0f, 1.0f, 1.0f);

		//Set the ray as alive
		rays[index].isTerminated = false;

	}

}


/**
*
*/
__global__ void pathIteration(int iter, RayState *rays, Camera *cam, Geom *geom, Material *mat, int geomCount, glm::vec3 *image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (x < cam->resolution.x && y < cam->resolution.y) {
		int index = x + (y * cam->resolution.x);
		
		if(!rays[index].isTerminated) {
			int intersectionT = -1;
			int materialIndex = 0;
			glm::vec3 intersectionPoint, intersectionNormal;
			for(int i = 0; i < geomCount; ++i) {
				glm::vec3 currentIntersectionPoint, currentNormal;
				bool outside = false;

				int t = -1;
				if(geom[i].type == GeomType::SPHERE) {
					t = sphereIntersectionTest(geom[i], rays[index].ray, currentIntersectionPoint, currentNormal, outside);
				} else if(geom[i].type == GeomType::CUBE) {
					t = boxIntersectionTest(geom[i], rays[index].ray, currentIntersectionPoint, currentNormal, outside);
				}

				if(t > 0 && (t < intersectionT || intersectionT < 0)) {
					materialIndex = geom[i].materialid;
					intersectionT = t;
					intersectionPoint = currentIntersectionPoint;
					intersectionNormal = currentNormal;
				}
			}

			if(intersectionT > 0) {
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);

				scatterRay(rays[index].ray, image[index], intersectionPoint, intersectionNormal, mat[materialIndex], rng);
			
				//Check if the geometry hit is a light source, set it as dead
				if(mat[materialIndex].emittance > 0) {
					rays[index].isTerminated = true;
				}

			} else {
				//The ray didn't intersect with anything, set it as dead
				rays[index].isTerminated = true;
			}
		}
	}
	
	
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(BLOCKSIZE, BLOCKSIZE);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray is a (ray, color) pair, where color starts as the
    //     multiplicative identity, white = (1, 1, 1).
    //   * For debugging, you can output your ray directions as colors.
    // * For each depth:
    //   * Compute one new (ray, color) pair along each path (using scatterRay).
    //     Note that many rays will terminate by hitting a light or hitting
    //     nothing at all. You'll have to decide how to represent your path rays
    //     and how you'll mark terminated rays.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //       surface.
    //     * You can debug your ray-scene intersections by displaying various
    //       values as colors, e.g., the first surface normal, the first bounced
    //       ray direction, the first unlit material color, etc.
    //   * Add all of the terminated rays' results into the appropriate pixels.
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    // * Finally, handle all of the paths that still haven't terminated.
    //   (Easy way is to make them black or background-colored.)

    //Generate all first level rays and save them 
	generateFirstLevelRays<<<blocksPerGrid2d, blockSize2d>>>(dev_camera, dev_ray_array);

	//Create a for loop that iterates over the desired depth
	//For each loop iteration 
	// * determine the number of threads and thus blocks needed
	// * call the pathtrace kernel for each ray
	// * do stream compaction to get rid of all the terminated rays and get the remaining number of rays!
	pathIteration<<<blocksPerGrid2d, blockSize2d>>>(iter, dev_ray_array, dev_camera, dev_scene_geom, dev_scene_material, hst_scene->geoms.size(), dev_image);
	checkCUDAError("path iteration");
    
	///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
	checkCUDAError("send to PBO");

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy image data");
}
