#ifndef MAIN_H
#define MAIN_H

/*
  Image Based Renderer

  Timothy Liu - 18945232
  U.C. Berkeley
  Electrical Engineering and Computer Sciences
  EE290T - Professor Avideh Zakhor

  main.h

*/

// SIFT Feature Detector by Rob Hess <hess@eecs.oregonstate.edu> http://web.engr.oregonstate.edu/~hess/
#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"

// OpenCV http://opencv.willowgarage.com/wiki/welcome
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

// OpenGL http://www.opengl.org
#include <GL/glu.h>
#include <GL/glut.h>

// Standard C header files
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <dirent.h>
#include <regex.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>

// DEBUG PRINT STATEMENTS
#define DEBUG 0

// the maximum number of keypoint NN candidates to check during BBF search
#define KDTREE_BBF_MAX_NN_CHKS 200

// threshold on squared ratio of distances between NN and 2nd NN
#define NN_SQ_DIST_RATIO_THR 0.25 //0.49

// regex for Image Files
#define IMG_REGEX "([^\\s]+(\\.(jpg|png|gif|bmp))$)"

// FILE CONSTANTS
#define MAX_FEATURE_THREADS 4
#define MAX_THREADS 24
#define MATCH_RANGE 3
#define NUM_IMAGES 261 // 261 // 784
#define STEP 3

#define OPTION 0
#define PI 3.14159265358979323846264338327950288
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800


// MACRO for max/min function
#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

// pose data struct
struct pData {
  CvScalar eye;
  CvScalar center;
  CvScalar up;
};

// scene data struct
struct scene {
  struct pData pose;
  int currentIndex;
  int previousIndex;
  int max_image;
  double xAngle;
  double yAngle;
  double zAngle;
};

// feature thread struct
struct fData {
  IplImage* img;
  int index;
};

// match thread struct
struct mData {
  IplImage* img1;
  IplImage* img2;
  int index1;
  int index2;
  CvMat* homography;
};

// ***************************CHECKPOINT 1 Methods: Initialization**************************
// Initialization
int initialize(FILE * poseFile, struct pData * poses, char ** filenames, int file_count);

// Initialization sort
int file_comp(const void * a, const void * b);

// ***************************CHECKPOINT 2 Methods: Loading Image Data**********************
// Loading Data
int loadImages(char* path, char** filenames, int file_count, IplImage** images);

// ***************************CHECKPOINT 3: Finding Features********************************
// Generate Features
void generateFeatures(IplImage ** allImages, int numImages);

// Feature Thread
void* featureThread(void* featureData);

// Print Features
void printFeature(struct feature *feat, int numFeatures, int step);

// ***************************CHECKPOINT 4: Calculating Homographies************************
// Enumerate Matches
int enumerateMatches(IplImage** images, int image_count, CvMat** matches);

// Match Thread
void* matchThread(void *matchData);

// Print Match/Homography/Matrix
void printMatrix(CvMat *matrix);

// ***************************CHECKPOINT 5: Calculating Norms of Homographies***************
// Calculate similarity of homographies
void calculateNorms(double* confidences, CvMat** matches, int image_count, int match_count);

// ***************************CHECKPOINT 6: Generating Best Homographies********************
// Search Function for Homographies
void generateBestHomographies(int* bestMatchedIndex, int image_count, double* confidences);

// ***************************Render Checkpoint: Rendering Scene****************************
// Init Default Scene
void initScene(int index, struct scene* aScene, struct pData* poses, int image_count);

// Find closest index to myScene's pose
int closestImageIndex(struct pData* poses);

// Calculate cameraHomography -- go from current position to camera position
CvMat* modelViewMatrix(int baseIndex, struct pData* poses);

// Render Scene Struct
void renderScene(IplImage** images, CvMat** matches, int* bestMatchedIndex, 
		 IplImage* viewport, struct pData* poses);

// Render more than one image per frame -- mosaic
void mosaic(int index, IplImage** images, CvMat** matches, int* bestMatchedIndex, 
		 IplImage* viewport, CvMat* initHomography);

// Update Scene Struct
void updateSceneKey(int key);

// Update Scene Struct
void updateSceneMouse(double diffX, double diffY);

// Print Scene Struct
void printScene(struct scene* aScene);

// Scene Mouse Event Handler
void mouseHandler(int event, int x, int y, int flags, void* param);

// ***************************Utility/Library Methods**************************************
// Temporary identity
CvMat* create3DIdentity(void);

CvMat* copyMatrix(CvMat* mat);

// Set a homography with values
void setHomography(CvMat *mat, double e00, double e01, double e02, 
		   double e10, double e11, double e12,
		   double e20, double e21, double e22);

// Print Pose Struct
void printPose(struct pData pose);

CvScalar createForwardVector(CvScalar *forward, 
			     struct pData dstPose, struct pData srcPose);

double dot(CvScalar one, CvScalar two);

double norm(CvScalar vec);

void makeXAxisRotation(CvMat* dst, double angle);
void makeYAxisRotation(CvMat* dst, double angle);
void makeZAxisRotation(CvMat* dst, double angle);

void projectTransform(CvMat* dst);

// absolute value for float
double absD(double d);

#endif // MAIN_H
