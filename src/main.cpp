/*
  Image Based Renderer

  by Timothy Liu
*/

// SIFT Feature Detector by Rob Hess <hess@eecs.oregonstate.edu> http://web.engr.oregonstate.edu/~hess/
#include <sift.h>
#include <imgfeatures.h>
#include <kdtree.h>
#include <utils.h>
#include <xform.h>

// OpenCV http://opencv.willowgarage.com/wiki/welcome
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

// OpenGL http://www.opengl.org/
#include <GL/glu.h>
#include <GL/glut.h>
#include "FreeImage.h"

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

// Standard C++ header files
#include <iostream>

using namespace std;

// the maximum number of keypoint NN candidates to check during BBF search
#define KDTREE_BBF_MAX_NN_CHKS 200

// threshold on squared ratio of distances between NN and 2nd NN
#define NN_SQ_DIST_RATIO_THR 0.25 //0.49

// regex for Image Files
#define IMG_REGEX "([^\\s]+(\\.(jpg|png|gif|bmp))$)"

// FILE CONSTANTS
#define MAX_FEATURE_THREADS 4
#define MAX_THREADS 24
#define NUM_IMAGES 784 // 261 // 784
#define RADIUS 1500
#define DEBUG 0

#define PI 3.14159265

// @TODO
struct Image {
  char filename[40];
  char time[40];
  CvScalar eye;
  CvScalar center;
  CvScalar up;
  struct Image** neighbors;
  int num_neighbors;
  int UL;
  int UR;
  int LL;
  int LR;
  double forwardAngle;
  double upAngleY;
  double upAngleX;
};

// feature thread struct
struct fData {
  IplImage* img;
  int index;
};

struct Plane {
  int num_points;
  CvScalar* points;
  double nx;
  double ny;
  double nz;
  double d;
};

struct Viewer {
  CvScalar eye;
  CvScalar center;
  CvScalar up;
  double xAngle;
  double yAngle;
  double zAngle;
  struct Image* view;
};

struct Viewport {
  int width;
  int height;
  int mouseX;
  int mouseY;
};

int initialize_poses(FILE * poseFile);
int initialize_planes(FILE * planeFile);
int intersect(CvScalar position, CvScalar direction, struct Plane* plane);
struct Image* getClosestImage();
void setHomography(struct Image* image);
double calculateHomography(struct Image* one, struct Image* two);
CvScalar createForwardVector(CvScalar eye, CvScalar center);
void renderScene(struct Image* image, GLuint textureID);

// ***************************CHECKPOINT 3: Finding Features********************************
// Generate Features
void generateFeatures(IplImage ** allImages, int numImages);

// Feature Thread
void* featureThread(void* featureData);

// Print Features
void printFeature(struct feature *feat, int numFeatures, int step);


/* Utility Functions */
double toDegrees(double radians);
void printImage(struct Image* image);
void printPlane(struct Plane* plane);
void printViewer(void);
void printScalar(CvScalar one);
CvScalar add(CvScalar one, CvScalar two);
CvScalar diff(CvScalar one, CvScalar two);
CvScalar normalize(CvScalar one);
double dist(CvScalar one, CvScalar two);
double dot(CvScalar one, CvScalar two);
void cleanup(void);

/* OpenGL functions */
static void initScene(int index);
static void resetScene(struct Image* image);
static void display(void);
static void reshape(int width, int height);
static void myKeyboardFunc(unsigned char key, int x, int y);
static void myPassiveMotionFunc(int x, int y);

void frameTimer(int value);

static char* PATH;
static Viewer* viewer;
static Viewport viewport;

static GLuint textTop[10];
static GLuint textTop2;

int frameCount = 0;
double velocity = 1;
int toggle = 1;
int file_count;
int num_planes;
struct Image* images;
struct Plane* planes;

int main(int argc, char** argv)
{
  cout << "START!" << endl;
  
  // PATH FOR IMAGE and POSE
  if (argc != 2) {
    printf("usage: %s <PATH_FOR_IMAGES>", argv[0]);
  }
  PATH = (char*) malloc(strlen(argv[1]));
  strcpy(PATH, argv[1]);


  // ************************TIME CHECKPOINT 1: Initialization****************************

  // Load PLANES;
  int c, count;
  char planeData[200];
  planes = NULL;

  // Parse Plane File
  FILE* planeFile;
  char planeFilename[strlen(PATH) + strlen("planes.model")];
  sprintf(planeFilename, "%splanes.model", PATH);
  
  planeFile = fopen(planeFilename, "r");
  if (planeFile != NULL) {

    // read in the number of planes
    count = 0;
    memset(planeData, 0, 200);
    while ((c = fgetc(planeFile)) != ' ' && c != '\n') {
      planeData[count] = (char) c;
      count++;
    }
    num_planes = atoi(planeData);
    
    planes = (struct Plane*) malloc(num_planes * sizeof(struct Plane));
  }
  initialize_planes(planeFile);

  // ************************TIME CHECKPOINT 2: Loading Image Data***************************

  // Load IMAGES
  file_count = 0;
  images = NULL;

  // Parse Pose File
  FILE* poseFile;
  char poseFilename[strlen(PATH) + strlen("poses.txt")];
  sprintf(poseFilename, "%sposes.txt", PATH);
  
  poseFile = fopen(poseFilename, "r");
  if (poseFile != NULL) {
    while (file_count < NUM_IMAGES) {
      images = (struct Image*) realloc(images, (file_count+1) * sizeof(struct Image));
      
      initialize_poses(poseFile);
      file_count++;
    } 

  }

  // Sort Images
  for (int i = 0; i < file_count; i++) {
    int image_count = 0;
    for (int j = 0; j < file_count; j++) {
      //if (i == j) continue;
      double distance = dist(images[i].eye, images[j].eye);
      //printf("distance: %.5lf\n", distance);

      if (distance < RADIUS) {

	// check for intersection
	if (i < j) {
	  int result;
	  for (int k = 0; k < num_planes; k++) {
	    result = intersect(images[i].eye, diff(images[j].center, images[i].eye), &planes[k]);
	    if (result == 1) {
	      //printf("intersect from image %d to image %d with plane %d\n", i, j, k);
	      //printPlane(&planes[k]);
	      continue;
	    }
	  }
	}

	images[i].neighbors = (struct Image**) realloc(images[i].neighbors, 
						      (image_count + 1) * sizeof(struct Image*));
	images[i].neighbors[image_count] = &images[j];
	image_count++;
	images[i].num_neighbors = image_count;
      }
    }
  }

  // ************************TIME CHECKPOINT 3: Finding Features*****************************

  /*

  pthread_t threads[file_count];
  struct fData thread_data[file_count];
  pthread_attr_t attr;
  void* status;

  int i = 0, rc, prevIndex1, prevIndex2, iteration_count;
  iteration_count = 1;
  prevIndex1 = 0;
  prevIndex2 = 0;
  while (i < numImages) {
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for (i = prevIndex1; i < min(numImages, MAX_FEATURE_THREADS*iteration_count); i++) {
      IplImage* img = cvCloneImage(allImages[i]);
      thread_data[i].img = img;
      thread_data[i].index = i;
	  
      if (DEBUG)
	printf("In generateFeatures: creating thread %ld\n", (long int) i);
      rc = pthread_create(&threads[i], &attr, featureThread, (void *) &thread_data[i]);
      if (rc){
	printf("ERROR; return code from pthread_create() is %d\n", rc);
	exit(-1);
      }
    }
    prevIndex1 = i;
    pthread_attr_destroy(&attr);
    for (i = prevIndex2; i < min(numImages, MAX_FEATURE_THREADS*iteration_count); i++) {
      rc = pthread_join(threads[i], &status);
      if (rc) {
	printf("ERROR; return code from pthread_join() is %d\n", rc);
	exit(-1);
      }
      cvReleaseImage(&thread_data[i].img);
    }
    prevIndex2 = i;
    iteration_count++;

  }

  */


  



  // Preprocess... calculate homographies
  for (int i = 0; i < file_count; i++) {
    setHomography(&images[i]);
  }

  if (DEBUG) {
    // PRINT PLANES
    for (int i = 0; i < num_planes; i++) {
      printf("*****Plane %d******\n", i+1);
      printPlane(&planes[i]);
    }

    // PRINT IMAGES
    for (int i = 0; i < file_count; i++) {
      printf("*****Image %d******\n", i+1);
      printImage(&images[i]);
    }
  }

  // Start OpenGL
  viewer = (struct Viewer*) malloc(sizeof(struct Viewer));
  viewer->eye = cvScalarAll(0.0);
  viewer->center = cvScalarAll(0.0);
  viewer->up = cvScalarAll(0.0);

  viewport.width = 1200;
  viewport.height = 1200;
  viewport.mouseX = 600;
  viewport.mouseY = 600;

  glutInit(&argc, argv);
  glutInitDisplayMode (GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

  // Create OpenGL Window
  glutInitWindowSize (1200, 1200);
  glutInitWindowPosition(0,0);
  glutCreateWindow("Image Based Renderer");

  //Register event handlers with OpenGL.
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(myKeyboardFunc);
  //glutSpecialFunc(specialKeyFunc); 
  //glutMotionFunc(myActiveMotionFunc);
  glutPassiveMotionFunc(myPassiveMotionFunc);
  frameTimer(0);

  initScene(10);

  glutMainLoop();

}

int initialize_planes(FILE* planeFile)
{
  int c, num_points, count;
  char planeData[200];

  for (int i = 0; i < num_planes; i++) {

    // parse the number of points
    count = 0;
    memset(planeData, 0, 200);
    while ((c = fgetc(planeFile)) != ' ' && c != '\n') {
      planeData[count] = (char) c;
      count++;
    }
    num_points = atoi(planeData);
    planes[i].num_points = num_points;

    planes[i].points = (CvScalar *) malloc(num_points * sizeof(CvScalar));

    // parse the plane equation
    for (int k = 0; k < 4; k++) {
      count = 0;
      memset(planeData, 0, 200);
      while ((c = fgetc(planeFile)) != ' ' && c != '\n') {
	planeData[count] = (char) c;
	count++;
      }
      if (k == 0)
	planes[i].nx = atof(planeData);
      else if (k == 1)
	planes[i].ny = atof(planeData);
      else if (k == 2)
	planes[i].nz = atof(planeData);
      else
	planes[i].d = atof(planeData);
    }

    // parse the plane extents
    for (int j = 0; j < num_points; j++) {
      for (int k = 0; k < 4; k++) {
	count = 0;
	memset(planeData, 0, 200);
	while ((c = fgetc(planeFile)) != ' ' && c != '\n') {
	  planeData[count] = (char) c;
	  count++;
	}
	planes[i].points[j].val[k] = atof(planeData);
      }
    }
    
    // parse the time stamp
    while ((c = fgetc(planeFile)) != '\n') {
    }
  }

}


int initialize_poses(FILE* poseFile)
{
  int c, count;
  char poseData[200];

  // filename
  count = 0;
  memset(images[file_count].filename, 0, 40);
  while((c = fgetc(poseFile)) != ' ') {
    images[file_count].filename[count] = (char) c;
    count++;
  }
  images[file_count].filename[count] = '.';
  images[file_count].filename[count+1] = 'j';
  images[file_count].filename[count+2] = 'p';
  images[file_count].filename[count+3] = 'g';

  // time
  count = 0;
  memset(images[file_count].time, 0, 40);
  while((c = fgetc(poseFile)) != ' ') {
    images[file_count].time[count] = (char) c;
    count++;
  }

  // center and up vector
  int matCount, vecCount;
  for (matCount = 0; matCount < 3; matCount++) {
    for (vecCount = 0; vecCount < 3; vecCount++) {
      count = 0;
      memset(poseData, 0, 200);
      while((c = fgetc(poseFile)) != ' ' && c != '\n') {
	poseData[count] = (char) c;
	count++;
      }
      if (matCount == 0) {
	images[file_count].eye.val[vecCount] = atof(poseData);
      } else if (matCount == 1) {
	images[file_count].center.val[vecCount] = atof(poseData);
      } else {
	images[file_count].up.val[vecCount] = atof(poseData);
      }
    }
  }

  // neighbors
  // TODO?
 
}

int intersect(CvScalar position, CvScalar direction, struct Plane* plane)
{
  double numerator = -1 * plane->d - plane->nx*position.val[0] - plane->ny*position.val[1] - plane->nz*position.val[2];
  double denominator = plane->nx*direction.val[0] + plane->ny*direction.val[1] + plane->nz*direction.val[2];
  
  if (denominator == 0) {
    return -1;
  } else {
    double t = numerator / denominator;
    //printf("intersection t = %.5f\n", t);

    if (t > 0 && t < 1) {
      return 1;
    } else {
      return 0;
    }
  }
}

struct Image* getClosestImage()
{
  // find the camera position closest to the viewer's eye
  double minDistance = 10000000000000000.0;
  struct Image* closestImage = NULL;
  for (int i = 0; i < file_count; i++) {
    double distance = dist(viewer->eye, images[i].eye);
    if (distance < minDistance) {
      minDistance = distance;
      closestImage = &images[i];
    }
  }

  // loop through all neighboring images from that camera's position
  // find the dot product between the viewer's direction and the image's direction
  // threshold the optimal looking images

  CvScalar image_direction;
  CvScalar viewer_direction = normalize(diff(viewer->center, viewer->eye));
  
  int count = 0;
  struct Image* bestNeighbors[closestImage->num_neighbors];
  for (int i = 0; i < closestImage->num_neighbors; i++) {

    image_direction = normalize(diff(closestImage->neighbors[i]->center,
    				      closestImage->neighbors[i]->eye));
  
    double directionalDistance = dot(image_direction, viewer_direction);

    if (directionalDistance > .95) {
      bestNeighbors[count] = closestImage->neighbors[i];
      count++;
    }

  }

  // find closest Image to viewer eye
  minDistance = 10000000000000000.0;
  closestImage = NULL;
  for (int i = 0; i < count; i++) {
    double distance = dist(viewer->eye, bestNeighbors[i]->eye);

    if (DEBUG) {
      printf("distance: %.2f\n", distance);
      printImage(bestNeighbors[i]);
    }

    if (distance < minDistance) {  
      minDistance = distance;
      closestImage = bestNeighbors[i];
    }
  }
  
  if (closestImage == NULL) {
    printf("*************NULL IMAGE***************\n");
    resetScene(viewer->view);
    return viewer->view;
  }

  return closestImage;

}


void setHomography(struct Image* image)
{
  // the z-axis
  
  CvScalar forwardVector = createForwardVector(image->eye, image->center);
  image->forwardAngle = atan2(-1*forwardVector.val[0], forwardVector.val[1]);
    
  CvScalar upVector = image->up;

  // the y-axis
  image->upAngleY = atan2(upVector.val[0], upVector.val[2]);
  
  // the x-axis
  image->upAngleX = atan2(upVector.val[2], upVector.val[1]);
  
  if (DEBUG) {
    printf("forwardAngle: %.2lf\n", toDegrees(image->forwardAngle));
    printf("upAngleY: %.2lf\n", toDegrees(image->upAngleY));
    printf("upAngleX: %.2lf\n\n", toDegrees(image->upAngleX));
  }

}




// homography from one to two
double calculateHomography(struct Image* one, struct Image* two)
{

  // the z-axis
  CvScalar forwardOne = diff(one->center, one->eye);
  CvScalar forwardTwo = diff(two->center, two->eye);
  CvScalar temp = cvScalarAll(0.0);
  double prod = dot(forwardOne, forwardTwo);
  double length = dist(forwardOne, temp);
  CvScalar projection = cvScalar(prod * forwardOne.val[0] / length, 
				 prod * forwardOne.val[1] / length,
				 prod * forwardOne.val[2] / length,
				 prod * forwardOne.val[3] / length);
  
  CvScalar norm = diff(forwardOne, projection);
 
  double forwardAngle = atan2(dist(norm, temp), dist(projection, temp));

  return forwardAngle;
}

CvScalar createForwardVector(CvScalar eye, CvScalar center)
{
  CvScalar forward = diff(center, eye);
  forward = normalize(forward);

  if (DEBUG)
    printf("forward: x = %.2lf, y = %.2lf, z = %.2lf\n", forward.val[0], forward.val[1], forward.val[2]);

  return forward;
}


// ***************************CHECKPOINT 3 Methods: Finding Features************************

// Generate Features - Step 1 of Algorithm

// Feature Thread
void* featureThread(void* featureData)
{
  char file[25];
  int index, numFeatures;
  struct feature* feat;
  
  struct fData* temp;
  temp = (struct fData*) featureData;
  IplImage* img = temp->img;
  index = temp->index;

  sprintf(file, "features/temp%d", index);  

  struct feature* feat1;
  if (import_features(file, FEATURE_LOWE, &feat1) == -1) {
    numFeatures = sift_features(img, &feat);
    export_features(file, feat, numFeatures);

      printf("features for image %d:\n", index);
    // print all features
    if (DEBUG) {

      printFeature(feat, numFeatures, 100);
      printf("\n\n");
    }

  }

}

// Print Features sampled at step
void printFeature(struct feature * feat, int numFeatures, int step) 
{
  struct feature* temp;
  int i;

  for (i = 0; i < numFeatures; i += step) {
    temp = feat + i;
    printf("feature %d located at (%f, %f) in image\n", i, temp->img_pt.x, temp->img_pt.y);
  }

}





/***********************
 *   UTILITY FUNCTIONS *
 * *********************/

double toDegrees(double radians)
{
  return radians * 180 / PI;
}

void printScalar(CvScalar one)
{
  printf("vector: (%.2f, %.2f, %.2f)\n",
	 one.val[0], one.val[1], one.val[2]);
}

void printImage(struct Image* image)
{
  printf("Filename: %s\n", image->filename);
  //printf("Timestamp: %s\n", image->time);
  printf("Eye Vector: (%.2f, %.2f, %.2f)\n",
	 image->eye.val[0], image->eye.val[1], image->eye.val[2]);
  printf("Center Vector: (%.2f, %.2f, %.2f)\n",
	 image->center.val[0], image->center.val[1], image->center.val[2]);
  /*
  printf("Up Vector: (%.2f, %.2f, %.2f)\n",
	 image->up.val[0], image->up.val[1], image->up.val[2]);
  */
  /*
    for (int i = 0; i < image->num_neighbors; i++) {
    printImage(&image->neighbors[i]);
    }
  */
  printf("Number of neighbors: %d\n", image->num_neighbors);
  cout << endl;
}

void printPlane(struct Plane* plane)
{
  printf("Plane nx = %.2f, ny = %.2f, nz = %.2f, d = %.2f\n", 
	 plane->nx, plane->ny, plane->nz, plane->d);
  for (int i = 0; i < plane->num_points; i++) {
    printScalar(plane->points[i]);
  }
}

void printViewer(void)
{
  printf("*******VIEWER********\n");
  printf("Viewer Eye Vector: (%.2f, %.2f, %.2f)\n",
	 viewer->eye.val[0], viewer->eye.val[1], viewer->eye.val[2]);
  printf("Viewer Center Vector: (%.2f, %.2f, %.2f)\n",
	 viewer->center.val[0], viewer->center.val[1], viewer->center.val[2]);
  printf("Viewer Up Vector: (%.2f, %.2f, %.2f)\n",
	 viewer->up.val[0], viewer->up.val[1], viewer->up.val[2]);
  printf("********VIEW*********\n");
  printImage(viewer->view);
}

double dist(CvScalar one, CvScalar two)
{
  double result = 0;
  result += pow(one.val[0]-two.val[0], 2);
  result += pow(one.val[1]-two.val[1], 2);
  result += pow(one.val[2]-two.val[2], 2);
  return sqrt(result);
}


CvScalar add(CvScalar one, CvScalar two)
{
  CvScalar temp = cvScalar(0.0, 0.0, 0.0, 0.0);
  temp.val[0] = one.val[0] + two.val[0];
  temp.val[1] = one.val[1] + two.val[1];
  temp.val[2] = one.val[2] + two.val[2];
  return temp;
}

CvScalar normalize(CvScalar one)
{
  CvScalar temp = cvScalarAll(0.0);
  double length = dist(one, temp);
  temp.val[0] = one.val[0] / length;
  temp.val[1] = one.val[1] / length;
  temp.val[2] = one.val[2] / length;
  return temp;
}

CvScalar diff(CvScalar one, CvScalar two)
{
  CvScalar temp = cvScalar(0.0, 0.0, 0.0, 0.0);
  temp.val[0] = one.val[0] - two.val[0];
  temp.val[1] = one.val[1] - two.val[1];
  temp.val[2] = one.val[2] - two.val[2];
  return temp;
}


double dot(CvScalar one, CvScalar two)
{
  double temp = 0.0;
  temp += one.val[0] * two.val[0];
  temp += one.val[1] * two.val[1];
  temp += one.val[2] * two.val[2];
  return temp;
}



/* OPENGL FUNCTIONS */


static void resetScene(struct Image* image)
{
  viewer->eye = image->eye;
  CvScalar temp_direction = diff(image->center, image->eye);
  viewer->center = add(image->eye, normalize(temp_direction));
  viewer->up = image->up;
  viewer->view = image;

  //printViewer();
}

static void initScene(int index)
{
  viewer->eye = images[index].eye;
  /*
  CvScalar temp_direction = diff(images[index].eye, images[index].center);
  viewer->center = add(images[index].eye, temp_direction);
  */
  
  CvScalar temp_direction = diff(images[index].center, images[index].eye);
  viewer->center = add(images[index].eye, normalize(temp_direction));
  
  /*
  viewer->center = images[index].center;
  */
  viewer->up = images[index].up;
  viewer->view = &images[index];

  //printViewer();

}

void renderScene(struct Image* image, GLuint textureID)
{
  if (textureID) {
    glDeleteTextures(1, &textureID);
  }
  
  char temp2[100];
  sprintf(temp2, "%s%s", PATH, image->filename);
  
  FREE_IMAGE_FORMAT img_format = FreeImage_GetFileType(temp2,0);
  FIBITMAP* bitmap = FreeImage_Load(img_format, temp2);
  FIBITMAP* temp = bitmap;
  
  bitmap = FreeImage_ConvertTo32Bits(bitmap);
  FreeImage_Unload(temp);
  
  int w = FreeImage_GetWidth(bitmap);
  int h = FreeImage_GetHeight(bitmap);  
  
  //    GLubyte* texture = new GLubyte[4*w*h];
  
  GLubyte* texture = (GLubyte* ) malloc(4*w*h);
  char* pixels = (char*)FreeImage_GetBits(bitmap);
  
  for(int j= 0; j<w*h; j++){
    texture[j*4+0]= pixels[j*4+2];
    texture[j*4+1]= pixels[j*4+1];
    texture[j*4+2]= pixels[j*4+0];
    texture[j*4+3]= pixels[j*4+3];
  }
  
  glClearColor (0.0, 0.0, 0.0, 0.0);
  glShadeModel(GL_FLAT);
  glEnable(GL_DEPTH_TEST);
  
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
  
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, 
		  GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
		  GL_LINEAR);
  
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, 
	       h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 
	       texture);  
  
  //printViewer();
  free(texture);
  FreeImage_Unload(bitmap);

  glEnable (GL_TEXTURE_2D); /* enable texture mapping */
    
  glBindTexture (GL_TEXTURE_2D, textureID); /* bind to our texture */
      
  glBegin (GL_QUADS);
    glTexCoord2f (0.0f,0.0f); /* lower left corner of image */
    glVertex3f (-134.5f, -100.7f, 0.0f);
    glTexCoord2f (1.0f, 0.0f); /* lower right corner of image */
    glVertex3f (134.5f, -100.7f, 0.0f);
    glTexCoord2f (1.0f, 1.0f); /* upper right corner of image */
    glVertex3f (134.5f, 100.7f, 0.0f);
    glTexCoord2f (0.0f, 1.0f); /* upper left corner of image */
    glVertex3f (-134.5f, 100.7f, 0.0f);
  glEnd ();

  glDisable (GL_TEXTURE_2D); /* disable texture mapping */

}

static void display(void)
{
    if (!viewport.height)
        return;

    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glMatrixMode (GL_MODELVIEW);
    glLoadIdentity ();
    glTranslatef (0, 0, -200); /* eye position */
        
    // find closest Image
    struct Image* closestImage = getClosestImage();
    if (viewer->view != closestImage) {      
      viewer->view = closestImage;
      viewer->eye = closestImage->eye;
      viewer->center = closestImage->center;      
    }

    // reset viewer to some global position
    //glRotatef(-113.8, 0, 1, 0);


    glPushMatrix(); // It is important to push the Matrix before calling glRotatef and glTranslatef
    
    // z-axis is y-axis
    glRotatef(toDegrees(viewer->view->forwardAngle), 0, 1, 0);
    
    // y-axis is x-axis
    glRotatef(-1 * toDegrees(viewer->view->upAngleY), 1, 0, 0);
    
    // x-axis is negative z-axis
    if (toDegrees(viewer->view->upAngleX) < 90) {
      glRotatef(90 - toDegrees(viewer->view->upAngleX), 0, 0, -1);
    } else {
      glRotatef(toDegrees(viewer->view->upAngleX) - 90, 0, 0, -1);
    }
    
    renderScene(viewer->view, textTop[0]);

    glPopMatrix();
    
    glutSwapBuffers();
    
    glutWarpPointer(600, 600);
    
    viewport.mouseX = 600;
    viewport.mouseY = 600;    	    

}

static void reshape (int width, int height)
{
  viewport.width = width;
  viewport.height = height;

  glViewport (0, 0, width, height);
  
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();
  
  gluPerspective (90, width / height, 1, 9999);
  
  glutPostRedisplay ();
}

static void myKeyboardFunc (unsigned char key, int x, int y)
{
  switch (key)
    {
      /* MOVEMENT */
      // LEFT
      case 'A':
      case 'a':
        viewer->eye.val[1] -= 10;
	viewer->center.val[1] -= 10;
	break;

      // RIGHT
      case 'D':
      case 'd':
	viewer->eye.val[1] += 10;
	viewer->center.val[1] += 10;
	break;

      // FORWARD
      case 'W':
      case 'w':
        viewer->eye.val[0] -= 10;
        viewer->center.val[0] -= 10;
	break;
	
      // BACK
      case 'S':
      case 's':
	viewer->eye.val[0] += 10;
	viewer->center.val[0] += 10;
	break;

      /* exit the program */
      case 27:
      case 'q':
      case 'Q':
	cleanup();
        exit (1);
        break;
    }
  //printViewer();
}

//-------------------------------------------------------------------------------
/// Called whenever the mouse moves without any buttons pressed.

static void myPassiveMotionFunc(int x, int y) 
{

  if (abs(x - viewport.mouseX) > 10) {
    double delta = PI * (abs(x - viewport.mouseX) - 10) / (viewport.width - 10);

    CvScalar difference = normalize(diff(viewer->center, viewer->eye));
    double angle = atan2(-1 * difference.val[0], difference.val[1]);

    CvScalar temp = cvScalarAll(0.0);
    double radius = dist(difference, temp);
    printf("radius: %.2f\n", radius);

    viewer->eye = cvScalar(viewer->eye.val[0] + radius * cos(angle + delta), 
				 viewer->eye.val[1] + radius * sin(angle + delta), 
				 viewer->eye.val[2], 0.0);			    
    printViewer();
  }

  glutPostRedisplay();
}


/// Called to update the screen at 30 fps.
void frameTimer(int value)
{
    frameCount += velocity;
    glutPostRedisplay();
    glutTimerFunc(1000/30, frameTimer, 1);
    //glutTimerFunc(1000/500, frameTimer, 1);
}


void cleanup(void)
{
  for (int i = 0; i < file_count; i++) {
    free(images[i].neighbors);
  }
  free(images);

  for (int i = 0; i < num_planes; i++) {
    free(planes[i].points);
  }
  free(planes);

  free(PATH);
  free(viewer);

  cout << "DONE!" << endl;
}
