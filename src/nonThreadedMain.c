/*
  Image Based Renderer

  Timothy Liu - 18945232
  U.C. Berkeley
  Electrical Engineering and Computer Sciences
  EE290T - Professor Avideh Zakhor
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
#include <sys/types.h>

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.25 //0.49

/* regex for Image Files */
#define IMG_REGEX "([^\\s]+(\\.(jpg|png|gif|bmp))$)"

#define DEBUG 0
#define IMAGE_LIMIT 50

/* OpenGL functions */
static void Init(void);
static void Draw(void);
static void Reshape(int width, int height);
static void Key(unsigned char key, int x, int y);

/* My image processing functions */
void generateFeatures(struct feature ** vector, int *countVector, IplImage ** allImages, int numImages);
void printFeature(struct feature *feat, int numFeatures, int step);
void enumerateComparisons(CvMat* combinations, IplImage ** allImages);
int enumerateMatches(CvMat** matches, CvPoint* matchIndices, int image_count, IplImage** images, 
		      struct feature** featuresVector, int* numFeaturesVector);
CvMat* create3DIdentity(void);
void  match(IplImage* img1, struct feature* feat1, int numFeatures1, IplImage* img2, 
	      struct feature* feat2, int numFeatures2, CvMat* homography);
void printMatrix(CvMat *matrix);


int generateBestHomographies(int* bestMatchedIndex, int* bestMatchedCounts, int image_count, 
			     double* confidences);

GLuint LoadTextureRAW(const char * filename, int wrap);

GLenum doubleBuffer;
GLint thing1, thing2;

struct mData {
  IplImage* img1;
  struct feature* feat1;
  int numFeatures1;
  IplImage* img2;
  struct feature* feat2;
  int numFeatures2;
  CvMat* homography;
};



// my main function
int main(int argc, char** argv)
{
  // OpenGL Declaration
  GLenum type;
  GLuint texture;

  // Loading Files
  regex_t re;
  DIR *directory;
  struct dirent *file;

  // Allocating data structures for data
  // init
  IplImage** images;
  // Step 1:
  struct feature** featuresVector;
  // Step 2:
  CvMat** matches;
  CvPoint* matchIndices;
  // Step 3
  double* confidences;
  // Step 4:
  int bestBaseImageIndex;
  long double bestTotal;
  double totalConfidence;
  double averageConfidence;

  // counts/lengths
  int* numFeaturesVector;
  int image_count, match_count;

  // Miscellaneous variables
  int i, j;
  CvMat* identity;
  time_t t1, t2, t3, t4, t5, t6, t7, renderTime, deallocateTime;

  // ***TIME CHECKPOINT 1: initialization***
  (void) time(&t1);
  printf("Starting Checkpoint 1 - initialization...\n");

  // OpenGL Init
  glutInit(&argc, argv);

  type = GLUT_RGB | GLUT_ACCUM;
  type |= (doubleBuffer) ? GLUT_DOUBLE : GLUT_SINGLE;
  glutInitDisplayMode(type);
  glutInitWindowSize(300, 300);
  glutCreateWindow("Hello World");

  // allocate a texture name
  texture = LoadTextureRAW("craptexture.PNG", 0);
  glEnable(GL_TEXTURE_2D);

  // OpenCV Init
  if (argc != 2) {
    fatal_error("usage: %s <PATH_FOR_IMAGES>", argv[0]);
  }

  // ***TIME CHECKPOINT 2: loading data***
  (void) time(&t2);
  printf("Starting Checkpoint 2 - loading images from directory...\n");

  // Find all images in directory
  if (regcomp(&re, IMG_REGEX, REG_EXTENDED|REG_NOSUB|REG_ICASE) != 0) {
    fatal_error("regex failed: %s", IMG_REGEX);
  }

  directory = opendir(argv[1]);
  if (!directory) {
    fatal_error("cannot find directory %s", argv[1]);
  }

  // Store valid images
  errno = 0;
  images = NULL;
  image_count = 0;
  while ((file = readdir(directory))) {
    // See if the file has image extension
    if (regexec(&re, file->d_name, (size_t) 0, NULL, 0) == 0) {
      //      if (DEBUG)
	printf("SUCCESSFUL IMAGE %d NAME: %s\n", image_count+1, file->d_name);
      
      // reallocate memory for dynamic array of images
      image_count++;
      images = (IplImage **) realloc(images, image_count * sizeof(IplImage *));
      
      // if reallocation fails: exit and free memory
      if (images == NULL) {
	fatal_error("(re)allocating memory when adding %s", file->d_name);
	for (i = 0; i < image_count; i++) {
	  cvReleaseImage(&images[i]);
	}
	free(images);
	regfree(&re);
	closedir(directory);
      }
      
      // load the image from file name
      char file_path[strlen(file->d_name) + strlen(argv[1])];
      sprintf(file_path, "%s%s", argv[1], file->d_name);
      images[image_count-1] = cvLoadImage(file_path, 1);

      // limit number of images
      if (image_count == IMAGE_LIMIT) break;

      // if loading images fail: exit and free memory        
      if (!images[image_count-1]) {
	fatal_error("cannot load image %s", file->d_name);
	for (i = 0; i < image_count-1; i++) {
	  cvReleaseImage(&images[i]);
	}
	free(images);
	regfree(&re);
	closedir(directory);
      }
    }
  }

  // if readdir fails, errno set to 1
  if (errno) {
    fatal_error("failed to read %s from directory", file->d_name);
    for (i = 0; i < image_count; i++) {
      cvReleaseImage(&images[i]);
    }
    free(images);
    regfree(&re);
    closedir(directory);
  }

  regfree(&re);
  closedir(directory);


  // ***TIME CHECKPOINT 3: Algorithm start Step 1***
  (void) time(&t3);
  printf("Starting Checkpoint 3 - finding features in each image...\n");
  
  /* insert algorithm A here*/
  // Step 1: generate features form images
  featuresVector = NULL;
  numFeaturesVector = NULL;

  featuresVector = (struct feature**) malloc(image_count * sizeof(struct feature *));
  numFeaturesVector = (int *) malloc(image_count * sizeof(int));
  generateFeatures(featuresVector, numFeaturesVector, images, image_count);

  // ***TIME CHECKPOINT 4: Algorithm start Step 2***
  (void) time(&t4);
  printf("Starting Checkpoint 4 - matching images...\n");

  // Step 2: enumerate all N^2 possible matches
  // match all images
  matches = NULL;
  matchIndices = NULL;

  matches = (CvMat **) malloc(image_count * image_count * sizeof(CvMat*));
  matchIndices = (CvPoint *) malloc(image_count * image_count * sizeof(CvPoint));
  match_count = enumerateMatches(matches, matchIndices, image_count, images, 
				 featuresVector, numFeaturesVector);

  // ***TIME CHECKPOINT 5: Algorithm start Step 3***
  (void) time(&t5);
  printf("Starting Checkpoint 5 - calculating norms of homography...\n");

  // Step 3: calculate norm of (homography - Identity Transformation)

  confidences = NULL;
  confidences = (double*) calloc(match_count, sizeof(double));
  // generateConfidences(confidences, matches, match_count, image_count);

  // create identity matrix
  identity = create3DIdentity();
  
  for (i = 0; i < match_count; i++) {
    if (!matches[i]) continue;
    // confidences[i] = cvNorm(matches[i], 0, CV_L2, 0);
    confidences[i] = cvNorm(matches[i], identity, CV_L2, 0);
  }
  if (DEBUG) {
    printf("\n");
    for (i = 0; i < image_count; i++) {
      for (j = 0; j < image_count; j++) {
	/*
	printf("confidence for matching image %d with image %d: %.2f\n", 
	       i, j, confidences[i*image_count + j]);
	*/
	printf("%.2f\t", confidences[i*image_count + j]);
      }
      printf("\n");
    }
  }

  // ***TIME CHECKPOINT 6: Algorithm start Step 5***
  (void) time(&t6);
  printf("Starting Checkpoint 6 - finding best homographies...\n");

  // Step 4: find the best homography for each image
  /* Algorithm 1 (correct):
   * create a priority queue for each image and its tranformations ranked by confidence
   * do a breadth first search outwards trying to reach the 'base' image
   * pick the shortest-weighted path to the base image, render by this sequence
   */
  /* Algorithm 2 (fast, mostly correct)
   * priority queue of length 1 for each image (i.e. the closest image)
   * keep track of the confidence between the image and the base image
   * traverse down to find path to base image, if path weight is higher than original, stop
   */

  int* bestMatchedIndex = (int*) calloc(image_count, sizeof(int));
  int* bestMatchedCounts = (int*) calloc(image_count, sizeof(int));

  bestBaseImageIndex = generateBestHomographies(bestMatchedIndex, bestMatchedCounts, 
						image_count, confidences);
  
  
  // ***TIME CHECKPOINT 7: Algorithm start Step 6
  (void) time(&t7);
  printf("Starting Checkpoint 7...\n");

  // Step 5: transform to world coordinates
  IplImage* finalImage = cvCreateImage(cvSize(1920, 1080), IPL_DEPTH_8U, 3);
  cvNamedWindow("Final Image", 1);

  int history[image_count];
  for (i = 0; i < image_count; i++) {
    history[image_count] = -1;
  }
  for (i = 0; i < image_count; i++) {
    //printf("displaying image %d\n", i+1);
    CvMat *ident = create3DIdentity();
    if (i == bestBaseImageIndex) {
      cvWarpPerspective(images[i], finalImage, ident, 0, cvScalarAll(0));
      continue;
    }
    CvMat *topHomography = matches[i*image_count + bestBaseImageIndex];
    double topConfidence = cvNorm(topHomography, ident, CV_L2, 0);

    CvMat* currHomography = matches[i*image_count + bestMatchedIndex[i]];
    int currIndex = bestMatchedIndex[i];
    history[i] = 1;
    history[currIndex] = 1;
    while (currIndex != bestBaseImageIndex) {
      CvMat* tempHomography = cvCreateMat(3, 3, CV_64F);
      cvMatMul(matches[currIndex*image_count + bestBaseImageIndex], 
	       currHomography, tempHomography);
      double currConfidence = cvNorm(tempHomography, ident, CV_L2, 0);
      if (currConfidence > topConfidence) {
	break;
      } else {
	cvMatMulAdd(matches[currIndex*image_count + bestMatchedIndex[currIndex]],
		    currHomography, 0, currHomography);
	currIndex = bestMatchedIndex[currIndex];
	if (history[currIndex] == 1) {
	  currIndex = -1;
	  break;
	}
	else history[currIndex] = 1;
      }
    
    }
    if (currIndex == bestBaseImageIndex) {
      if (DEBUG) {
	printf("Index to Match: bestBaseImageIndex = %d\n", currIndex);
	printf("Chosen homography: ");
	printMatrix(currHomography);
      }
      
      if (cvNorm(currHomography, 0, CV_L2, 0) > 1000) continue;
      cvWarpPerspective(images[i], finalImage, currHomography, 0, cvScalarAll(0));
    } else {
      if (DEBUG) {
	printf("Index to Match: currIndex = %d\n", currIndex);
	printf("Chosen homography: ");
	printMatrix(topHomography);
      }

      if (cvNorm(topHomography, 0, CV_L2, 0) > 1000) continue;
      cvWarpPerspective(images[i], finalImage, topHomography, 0, cvScalarAll(0));
    }
    if (DEBUG){
      cvShowImage("Final Image", finalImage);
      cvWaitKey(0);
    }
  }
  cvShowImage("Final Image", finalImage);
  cvWaitKey(0);
  cvReleaseImage(&finalImage);
  cvReleaseMat(&identity);

  // ***TIME CHECKPOINT: OpenGL Render***
  (void) time(&renderTime);

  Init();

  // ***DEALLOCATION CHECKPOINT: Deallocating Memory***
  (void) time(&deallocateTime);

  // free allocated data
  // free step 3 & 4
  free(confidences);

  // free step 2
  for (i = 0; i < match_count; i++) {
    cvReleaseMat(&matches[i]);
  }
  free(matches);
  free(matchIndices);

  // free init and step 1
  for (i = 0; i < image_count; i++) {
    free(featuresVector[i]);
    cvReleaseImage(&images[i]);
  }
  free(featuresVector);
  free(numFeaturesVector);
  free(images);

  // Print Statistics
  printf("\n");
  printf("Number of images: %d\n", image_count);
  printf("Time for initialization: %.2lf\n", difftime(t2,t1));
  printf("Time to load data: %.2lf\n", difftime(t3,t2));
  printf("Time to generate features: %.2lf\n", difftime(t4,t3));
  printf("Time to generate matches: %.2lf\n", difftime(t5,t4));
  printf("Time to generate confidence/norms: %.2lf\n", difftime(t6, t5));
  printf("Time to search for optimal homographies: %.2lf\n", difftime(t7, t6));
  printf("Time to transform to world coordinates: %.2lf\n", difftime(renderTime, t7));
  printf("Time for rendering: %.2lf\n", difftime(deallocateTime,renderTime));
  printf("Total Time: %.2lf\n", difftime(deallocateTime, t1));
   
  glutReshapeFunc(Reshape);
  glutKeyboardFunc(Key);
  glutDisplayFunc(Draw);
  glutMainLoop();

  return 0;

}

CvMat* create3DIdentity(void) 
{
  int i, j;
  CvMat *temp = cvCreateMat(3, 3, CV_64F);
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      if (i == j) cvmSet(temp, i, j, 1.0);
      else cvmSet(temp, i, j, 0.0);
    }
  }
  return temp;
}


// Generate Features - Step 1 of Algorithm
void generateFeatures(struct feature** vector, int* countVector, IplImage ** allImages, int numImages) 
{
  int i, numFeatures;
  struct feature* feat;

  // find features for image at index i and store in array vector
  for (i = 0; i < numImages; i++) {
    if (DEBUG) printf("finding features for image %d...", i+1);
    numFeatures = sift_features(allImages[i], &feat);
    if (DEBUG) printf("%d features found!\n", numFeatures);
    vector[i] = feat;
    countVector[i] = numFeatures;
  }

  // print all features
  if (DEBUG) {
    for (i = 0; i < numImages; i++) {
      printf("features for image %d:\n", i+1);
      printFeature(vector[i], countVector[i], 100);
      printf("\n\n");
    }
  }

}

// helper function
void printFeature(struct feature * feat, int numFeatures, int step) 
{
  struct feature* temp;
  int i;

  for (i = 0; i < numFeatures; i += step) {
    temp = feat + i;
    printf("feature %d located at (%f, %f) in image\n", i+1, temp->img_pt.x, temp->img_pt.y);
  }

}

// Enumerate all combinations of Matches - Step 2 of Algorithm
int enumerateMatches(CvMat** matches, CvPoint* matchIndices, int image_count, IplImage** images, 
		      struct feature** featuresVector, int* numFeaturesVector)
{
  int i, j, x, y, index, rc, match_count = 0;

  for (i = 0; i < image_count; i++) {
    for (j = 0; j < image_count; j++) {
      printf("matching image %d with image %d\n", i+1, j+1);
      index = i*image_count+j;

      if (i == j) {
	matches[index] = cvCreateMat(3, 3, CV_64F);
	for (x = 0; x < 3; x++) {
	  for (y = 0; y < 3; y++) {
	    cvmSet(matches[index], x, y, 1000.0);
	  }
	}
      } else if (i > j) {
	CvMat* inv = cvCreateMat(3, 3, CV_64F);
	cvInvert(matches[j*image_count + i], inv, CV_LU);
	matches[index] = inv;
      } else {
	matches[index] = cvCreateMat(3, 3, CV_64F);
	match(images[i], featuresVector[i], numFeaturesVector[i],
	      images[j], featuresVector[j], numFeaturesVector[j], 
	      matches[i*image_count+j]);
	printMatrix(matches[index]);
      }
      matchIndices[index] = cvPoint(i, j);
      match_count++;
    }
  }

  if (DEBUG) {
    for (i = 0; i < match_count; i++) {
      printf("homography for match %d\n", i+1);
      printMatrix(matches[i]);
    }
  }
    
  return match_count;

}

void match(IplImage* img1, struct feature* feat1, int numFeatures1, IplImage* img2, struct feature* feat2, int numFeatures2, CvMat * homography)
{

  IplImage* stacked;
  struct feature* feat;
  struct feature** nbrs;
  struct kd_node* kd_root;
  CvPoint pt1, pt2;
  double d0, d1;
  int k, i, m = 0;

  stacked = stack_imgs( img1, img2 );

  kd_root = kdtree_build( feat2, numFeatures2 );
  for( i = 0; i < numFeatures1; i++ ) {
      feat = feat1 + i;
      k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
      if( k == 2 ) {
	  d0 = descr_dist_sq( feat, nbrs[0] );
	  d1 = descr_dist_sq( feat, nbrs[1] );
	  if( d0 < d1 * NN_SQ_DIST_RATIO_THR ) {
	      pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
	      pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
	      pt2.y += img1->height;
	      cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
	      //printf("absolute distance: %f\n", sqrt(pow((pt1.x - pt2.x), 2) + pow((pt1.y - pt2.y), 2)));
	      m++;
	      feat1[i].fwd_match = nbrs[0];
	  }
      }
      free( nbrs );
  }

  char blegh[15];
  sprintf(blegh, "features.png");

  //  if (DEBUG) {
    printf("Found %d total matches\n", m );
    display_big_img( stacked, "Matches" );
    cvSaveImage(blegh, stacked);
    cvWaitKey( 0 );
    //}


  CvMat* H;

  H = ransac_xform(feat1, numFeatures1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
		   homog_xfer_err, 3.0, NULL, NULL);

  if (H) {
    /*
    //IplImage *xformed = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);
    IplImage *xformed = cvCloneImage(img2);
    //cvWarpPerspective(img1, xformed, H, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
    cvWarpPerspective(img1, xformed, H, 0, cvScalarAll(0));
    cvNamedWindow("TRANSFORMED IMAGE", 1);
    cvShowImage("TRANSFORMED IMAGE", xformed);
    cvWaitKey(0);
    cvReleaseImage(&xformed);
    //*/
  } else {
    H = cvCreateMat(3, 3, CV_64F);
    cvmSet(H, 0, 0, 1000.0);
    cvmSet(H, 0, 1, 1000.0);
    cvmSet(H, 0, 2, 1000.0);
    cvmSet(H, 1, 0, 1000.0);
    cvmSet(H, 1, 1, 1000.0);
    cvmSet(H, 1, 2, 1000.0);
    cvmSet(H, 2, 0, 1000.0);
    cvmSet(H, 2, 1, 1000.0);
    cvmSet(H, 2, 2, 1000.0);
  }

  cvReleaseImage(&stacked);
  kdtree_release(kd_root);
  //printMatrix(H);
  int x, y;
  for (x = 0; x < 3; x++) {
    for (y = 0; y < 3; y++) {
      cvmSet(homography, x, y, cvmGet(H, x, y));
    }
  }
  //homography = H;
  //return H;
  
}

void printMatrix(CvMat *matrix) 
{
  int i, j;
  for (i = 0; i < matrix->rows; i++) {
    for (j = 0; j < matrix->cols; j++) {
      printf("%8.3f ", (float) cvGetReal2D(matrix, i, j));
    }
    printf("\n");
  }
}

// Find best homography - Step 4 of Algorithm:
int generateBestHomographies(int* bestMatchedIndex, int* bestMatchedCounts, int image_count, 
			     double* confidences) 
{
  int i, j, minConfidenceIndex;
  long double minConfidence;
  for (i = 0; i < image_count; i++) {
    minConfidence = 100000000000000000.0;
    minConfidenceIndex = 0;
    for (j = 0; j < image_count; j++) {
      if (i == j) continue;
      double value = confidences[i*image_count + j];
      if (value < minConfidence) {
	minConfidence = value;
	minConfidenceIndex = j;
      }
    }
    bestMatchedIndex[i] = minConfidenceIndex;
    bestMatchedCounts[minConfidenceIndex]++;
  }

  int modeValue = 0;
  int maxCounts = 0;
  for (i = 0; i < image_count; i++) {
    //    if (DEBUG) 
      printf("Best Match for image %d is image %d\n", i+1, bestMatchedIndex[i]+1);
    if (bestMatchedCounts[i] > maxCounts) {
      maxCounts = bestMatchedCounts[i];
      modeValue = i;
    }
  }
  //  if (DEBUG) 
    printf("Best Base Image Index: %d\n", modeValue+1);

  return modeValue;

}
static void Init(void)
{

  int j;

  LoadTextureRAW("craptexture.PNG", 0);
 
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glShadeModel(GL_FLAT);
  glEnable(GL_DEPTH_TEST);
  glClearAccum(0.0, 0.0, 0.0, 0.0);
  
  thing1 = glGenLists(1);
  glNewList(thing1, GL_COMPILE);
  glColor3f(1.0, 0.0, 0.0);
  glRectf(-1.0, -1.0, 1.0, 0.0);
  glEndList();
  
  thing2 = glGenLists(1);
  glNewList(thing2, GL_COMPILE);
  glColor3f(0.0, 1.0, 0.0);
  glRectf(0.0, -1.0, 1.0, 1.0);
  glEndList();
}

static void Reshape(int width, int height)
{
  
  glViewport(0, 0, width, height);
  
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

static void Key(unsigned char key, int x, int y)
{
  
  switch (key) {
  case '1':
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glutPostRedisplay();
    break;
  case '2':
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glutPostRedisplay();
    break;
  case 27:
    exit(0);
  }
}

static void Draw(void)
{
  
  glPushMatrix();
  
  glScalef(0.8, 0.8, 1.0);
  
  glClear(GL_COLOR_BUFFER_BIT);
  glCallList(thing1);
  glAccum(GL_LOAD, 0.5);
  
  glClear(GL_COLOR_BUFFER_BIT);
  glCallList(thing2);
  glAccum(GL_ACCUM, 0.5);
  
  glAccum(GL_RETURN, 1.0);

  glBegin( GL_QUADS );
  glTexCoord2d(0.0,0.0); glVertex2d(0.0,0.0);
  glTexCoord2d(1.0,0.0); glVertex2d(1.0,0.0);
  glTexCoord2d(1.0,1.0); glVertex2d(1.0,1.0);
  glTexCoord2d(0.0,1.0); glVertex2d(0.0,1.0);
  glEnd();
  
  glPopMatrix();
  
  if (doubleBuffer) {
    glutSwapBuffers();
  } else {
    glFlush();
  }
}

GLuint LoadTextureRAW( const char * filename, int wrap )
{
    GLuint texture;
    int width, height;
    BYTE * data;
    FILE * file;

    // open texture data
    file = fopen( filename, "rb" );
    if ( file == NULL ) return 0;

    // allocate buffer
    width = 256;
    height = 256;
    data = malloc( width * height * 3 );

    // read texture data
    fread( data, width * height * 3, 1, file );
    fclose( file );

    // allocate a texture name
    glGenTextures( 1, &texture );

    // select our current texture
    glBindTexture( GL_TEXTURE_2D, texture );

    // select modulate to mix texture with color for shading
    glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

    // when texture area is small, bilinear filter the closest mipmap
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                     GL_LINEAR_MIPMAP_NEAREST );
    // when texture area is large, bilinear filter the first mipmap
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    // if wrap is true, the texture wraps over at the edges (repeat)
    //       ... false, the texture ends at the edges (clamp)
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                     wrap ? GL_REPEAT : GL_CLAMP );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                     wrap ? GL_REPEAT : GL_CLAMP );

    // build our texture mipmaps
    gluBuild2DMipmaps( GL_TEXTURE_2D, 3, width, height,
                       GL_RGB, GL_UNSIGNED_BYTE, data );

    // free buffer
    free( data );

    return texture;
}
