/*
  Image Based Renderer

  Timothy Liu - 18945232
  U.C. Berkeley
  Electrical Engineering and Computer Sciences
  EE290T - Professor Avideh Zakhor

  main.c
*/

#include "main.h"

struct scene* myScene;
int leftClicked;
int rightClicked;
int middleClicked;
double mousePosX;
double mousePosY;

// my main function
int main(int argc, char** argv)
{
  // Main Variables
  int i, j;
  time_t t1, t2, t3, t4, t5, t6, t7, renderTime, deallocateTime;

  if (argc != 2) {
    fatal_error("usage: %s <PATH_FOR_IMAGES>", argv[0]);
  }

  // ************************TIME CHECKPOINT 1: Initialization****************************
  (void) time(&t1);
  printf("Starting Checkpoint 1 - initialization...\n");

  // Variables
  int file_count = 0;
  char** filenames = NULL;
  struct pData* poses = NULL;

  FILE *poseFile;
  char poseFilename[strlen(argv[1]) + strlen("poses.txt")];
  sprintf(poseFilename, "%sposes.txt", argv[1]);
    
  poseFile = fopen(poseFilename, "r");

  if (poseFile != NULL) {
    int c, step = 0;    
    while (file_count < NUM_IMAGES) {
      filenames = (char **) realloc(filenames, (file_count+1) * sizeof(char *));
      poses = (struct pData*) realloc(poses, (file_count+1) * sizeof(struct pData));

      if ((step % STEP) == 0) {
	initialize(poseFile, poses, filenames, file_count);
	file_count++;
	step++;
      } else {
	while((c = fgetc(poseFile)) != '\n');
	step++;
      }
    }      
  }

  // ************************TIME CHECKPOINT 2: Loading Image Data***************************
  (void) time(&t2);
  printf("Starting Checkpoint 2 - loading images from directory...\n");

  // Variables
  int image_count = 0;
  IplImage** images = NULL;

  images = (IplImage**) malloc(file_count * sizeof(IplImage*));
  // ERROR check allocation
  if (images == NULL) {
    fatal_error("allocating memory for images");
  }

  // load images
  memset(images, 0, file_count*sizeof(IplImage*));  
  image_count = loadImages(argv[1], filenames, file_count, images);

  // Release the memory for filenames.  Not needed anymore.
  free(filenames);

  // ************************TIME CHECKPOINT 3: Finding Features*****************************
  (void) time(&t3);
  printf("Starting Checkpoint 3 - finding features in each image...\n");

  // Step 1: generate features form images
  generateFeatures(images, image_count);

  // ************************TIME CHECKPOINT 4: Calculating Homographies*********************
  (void) time(&t4);
  printf("Starting Checkpoint 4 - matching images...\n");

  // A match represents a homography between two images
  int match_count = 0;
  CvMat** matches = (CvMat **) malloc(image_count * image_count * sizeof(CvMat*));

  // ERROR check allocation
  if (matches == NULL) {
    fatal_error("allocating memory for matches");
  }

  // Calculate homographies
  memset(matches, 0, image_count*image_count*sizeof(CvMat*));  
  match_count = enumerateMatches(images, image_count, matches);

  // ************************TIME CHECKPOINT 5: Calculating Norms of Homographies************
  (void) time(&t5);
  printf("Starting Checkpoint 5 - calculating norms of homography...\n");

  // The confidence is a measurement of similarity between images (i.e. norm of homography)
  double* confidences = (double*) calloc(match_count, sizeof(double));

  // ERROR check allocation
  if (confidences == NULL) {
    fatal_error("allocating memory for confidences");
  }
  
  calculateNorms(confidences, matches, image_count, match_count);

  // ************************TIME CHECKPOINT 6: Generating Best Homographies************
  (void) time(&t6);
  printf("Starting Checkpoint 6 - Generating best homographies...\n");

  // Variables
  int* bestMatchedIndex = (int*) calloc(image_count, sizeof(int));
  generateBestHomographies(bestMatchedIndex, image_count, confidences);

  // ************************Render Checkpoint: Rendering Scene****************************
  (void) time(&renderTime);
  printf("Starting Render Checkpoint - Rendering...\n");

  // Variables
  IplImage* viewport;
  clock_t currtime;
  int key;
  
  myScene = (struct scene*) malloc(sizeof(struct scene));
  initScene(0, myScene, poses, image_count);

  cvNamedWindow("Scene", 1);
  cvMoveWindow("Scene", 560, 200);

  rightClicked = 0;
  leftClicked = 0;
  middleClicked = 0;
  mousePosX = 0;
  mousePosY = 0;
  cvSetMouseCallback("Scene", mouseHandler, 0);
  viewport = cvCreateImage(cvSize(WINDOW_WIDTH, WINDOW_HEIGHT), IPL_DEPTH_8U, 3);
  
  while(1) {
    currtime = clock();

    if (!OPTION) {
      cvSetZero(viewport);
      //cvReleaseImage(&viewport);
      //viewport = cvCreateImage(cvSize(WINDOW_WIDTH, WINDOW_HEIGHT), IPL_DEPTH_8U, 3);
    }

    renderScene(images, matches, bestMatchedIndex, viewport, poses);
    cvShowImage("Scene", viewport);

    key = cvWaitKey(10);
    if (key == 'e' || key == 'E' || key == 27) {
      printf("EXITING!!!\n");
      break;
    } else if (key == 'r' || key == 'R') {
      initScene(0, myScene, poses, image_count);
    }else if (OPTION) {
      updateSceneKey(key);
    } 

    //if (DEBUG) 
      printf("time for frame: %.2lf\n", (((double) 1.0*clock()) - currtime) / CLOCKS_PER_SEC);
  }
  cvReleaseImage(&viewport);
  
  // ************************DEALLOCATION CHECKPOINT: Deallocating Memory******************
  (void) time(&deallocateTime);
  printf("Starting Deallocation Checkpoint - Deallocating...\n");

  // free allocated data 
  // free CHECKPOINT 7
  free(myScene);

  // free CHECKPOINT 6
  free(bestMatchedIndex);

  // free CHECKPOINT 5
  free(confidences);

  // free CHECKPOINT 3 and 4
  for (i = 0; i < match_count; i++) {
    cvReleaseMat(&matches[i]);
  }
  free(matches);  
  
  // free CHECKPOINT 2
  for (i = 0; i < image_count; i++) {
    cvReleaseImage(&images[i]);
  }
  free(images);

  // free CHECKPOINT 1
  free(poses);
 
  // Print Statistics
  printf("\n");
  printf("Number of images: %d\n", image_count);
  printf("Time for initialization: %.2lf\n", difftime(t2,t1));
  printf("Time to load data: %.2lf\n", difftime(t3,t2));
  printf("Time to generate features: %.2lf\n", difftime(t4,t3));
  printf("Time to generate matches: %.2lf\n", difftime(t5,t4));
  printf("Time to generate confidence/norms: %.2lf\n", difftime(t6, t5));
  printf("Time to search for optimal homographies: %.2lf\n", difftime(renderTime, t6));
  printf("Time for rendering: %.2lf\n", difftime(deallocateTime,renderTime));
  printf("Total Time: %.2lf\n", difftime(deallocateTime, t1));
     
  return 0;

}

// ***************************CHECKPOINT 1 Methods: Initialization**************************
// Initialization
int initialize(FILE * poseFile, struct pData * poses, char ** filenames, int file_count)
{
  int c, count, vecCount, matCount;
  char filename[45];
  char timeData[40];
  char poseData[200];

  poses[file_count].eye = cvScalarAll(0.0);
  poses[file_count].center = cvScalarAll(0.0);
  poses[file_count].up = cvScalarAll(0.0);
  
  count = 0;
  memset(filename, 0, 40);
  while((c = fgetc(poseFile)) != ' ') {
    filename[count] = (char) c;
    count++;
  }
  filename[count] = '.';
  filename[count+1] = 'j';
  filename[count+2] = 'p';
  filename[count+3] = 'g';
  
  filenames[file_count] = (char*) malloc((strlen(filename) + 4) * sizeof(char));
  strcpy(filenames[file_count], filename);
  
  count = 0;
  memset(timeData, 0, 40);
  while((c = fgetc(poseFile)) != ' ') {
    timeData[count] = (char) c;
    count++;
  }
  
  for (matCount = 0; matCount < 3; matCount++) {
    for (vecCount = 0; vecCount < 3; vecCount++) {
      count = 0;
      memset(poseData, 0, 200);
      while((c = fgetc(poseFile)) != ' ' && c != '\n') {
	poseData[count] = (char) c;
	count++;
      }
      if (matCount == 0) {
	poses[file_count].eye.val[vecCount] = atof(poseData);
      } else if (matCount == 1) {
	poses[file_count].center.val[vecCount] = atof(poseData);
      } else {
	poses[file_count].up.val[vecCount] = atof(poseData);
      }
    }
  }
  
}

// Initialization sort
int file_comp(const void * a, const void * b)
{
  return strcmp(*(char**)a, *(char**)b);
}


// ***************************CHECKPOINT 2 Methods: Loading Image Data**********************
// Load Images from path
int loadImages(char* path, char** filenames, int file_count, IplImage** images)
{
  if (DEBUG)
    printf("file_count: %d\n", file_count);
  int i, j;

  for (i = 0; i < file_count; i++) {
    //printf("file named: %s\n", filenames[i]);
    // load the image from file name
    char file_path[strlen(filenames[i]) + strlen(path)];
    sprintf(file_path, "%s%s", path, filenames[i]);
    if (DEBUG)
      printf("path to load from: %s\n", file_path);
    images[i] = cvLoadImage(file_path, 1);

    // if loading images fail: exit and free memory        
    if (!images[i]) {
      fatal_error("cannot load image %s", file_path);
      for (j = 0; j < i; j++) {
	cvReleaseImage(&images[j]);
      }
      free(images);
      free(filenames);
    }
  }

  return file_count;
}

// ***************************CHECKPOINT 3 Methods: Finding Features************************

// Generate Features - Step 1 of Algorithm
void generateFeatures(IplImage ** allImages, int numImages) 
{
  pthread_t threads[numImages];
  struct fData thread_data[numImages];
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
}

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

// ***************************CHECKPOINT 4 Methods: Calculating Homographies****************
// Enumerate all combinations of Matches - Step 2 of Algorithm
int enumerateMatches(IplImage** images, int image_count, CvMat** matches)
{
  //printf("enumerating matches...\n");
  int i, j, x, y, index, prevIndex, prevIndex2, rc;
  int match_count = 0, iteration_count = 0, range_count = 0;

  pthread_t threads[image_count*image_count];
  struct mData thread_data[image_count*image_count];
  pthread_attr_t attr;
  void* status;

  for (i = 0; i < MATCH_RANGE; i++) {
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    j = 0;
    // traversing diagonally down to avoid collisions
    prevIndex = i+1;
    prevIndex2 = i+1;
    iteration_count = 1;
    while (j < i + image_count) {
      
      for (j = prevIndex; j < min(i+image_count+1, i+MAX_THREADS*iteration_count); j++) {
	//printf("j = %d | iteration_count = %d\n", j, iteration_count);
	x = (j - i - 1) % image_count;
	if (x < 0) x += image_count;
	y = j % image_count;
	if (y < 0) y += image_count;
	
	//printf("matching image %d with image %d\n", x, y);
	index = x*image_count+y;

	matches[index] = cvCreateMat(3, 3, CV_64F);
	
	IplImage* clone1 = cvCloneImage(images[x]);
	IplImage* clone2 = cvCloneImage(images[y]);
	
	thread_data[index].img1 = clone1;
	thread_data[index].index1 = x;
	
	thread_data[index].img2 = clone2;
	thread_data[index].index2 = y;
	
	thread_data[index].homography = matches[index];
	
	if (DEBUG)
	  printf("In enumerateMatches: creating thread %ld\n", (long int) (index));
	rc = pthread_create(&threads[index], &attr, matchThread, (void *) &thread_data[index]);
	if (rc){
	  printf("ERROR; return code from pthread_create() is %d\n", rc);
	  exit(-1);
	}
	cvReleaseImage(&clone1);
	cvReleaseImage(&clone2);
	
      }
      prevIndex = j;

      pthread_attr_destroy(&attr);
      
      for (j = prevIndex2; j < min(i+image_count+1, i+MAX_THREADS*iteration_count); j++) {
	x = (j - i - 1) % image_count;
	if (x < 0) x += image_count;
	y = j % image_count;
	if (y < 0) y += image_count;
	index = x*image_count + y;
	
	rc = pthread_join(threads[index], &status);
	if (rc) {
	  printf("ERROR; return code from pthread_join() is %d\n", rc);
	  exit(-1);
	}
	if (DEBUG) {
	  printf("Main: completed join with thread %ld having a status of %ld\n", 
		 (long int) index ,(long)status);
	  printMatrix(matches[index]);
	}
      }
      prevIndex2 = j;
      iteration_count++;
    }
  }

  range_count = 0;
  for (i = 0; i < image_count; i++) {    
    for (j = 0; j < image_count; j++) {
      index = i*image_count + j;
      if (i == j) {
	matches[index] = cvCreateMat(3, 3, CV_64F);
	for (x = 0; x < 3; x++) {
	  for (y = 0; y < 3; y++) {
	    cvmSet(matches[index], x, y, 1000000.0);
	  }
	}
      } else if (i > j) {
	if (matches[j*image_count+i]) {
	  CvMat* inv = cvCreateMat(3, 3, CV_64F);
	  cvInvert(matches[j*image_count + i], inv, CV_LU);
	  matches[index] = inv;
	}
      } else {
	if (matches[j*image_count+i] && !matches[index]) {
	  CvMat* inv = cvCreateMat(3, 3, CV_64F);
	  cvInvert(matches[j*image_count + i], inv, CV_LU);
	  matches[index] = inv;
	}
      }
      match_count++;
    }
  }

  if (DEBUG) {
  
    for (i = 0; i < match_count; i++) {
      if (matches[i]) {
	printf("homography for match %d\n", i);
	printMatrix(matches[i]);
      }
    }
  
  }

  if (DEBUG)
    printf("done with generateMatches\n");

  return match_count;

}

// Match Thread
void* matchThread(void* matchData)
{
  // normal match local variables
  struct feature* feat= NULL;
  struct feature** nbrs = NULL;
  struct kd_node* kd_root = NULL;
  double d0, d1;
  int k, i, x, y, m = 0;

  // arguments passed to thread
  char file[25];
  char file2[25];
  struct mData* temp;
  temp = (struct mData*) matchData;

  IplImage* img1 = temp->img1;
  IplImage* img2 = temp->img2;
  int index1 = temp->index1;
  int index2 = temp->index2;
  CvMat* homography = temp->homography;

  sprintf(file, "features/temp%d", index1);
  sprintf(file2, "features/temp%d", index2);

  struct feature* feat1;
  struct feature* feat2;
  
  int numFeatures1 = import_features(file, FEATURE_LOWE, &feat1);
  int numFeatures2 = import_features(file2, FEATURE_LOWE, &feat2);

  // printf("running the matching function\n");

  // matching function
  kd_root = kdtree_build( feat2, numFeatures2 );
  for( i = 0; i < numFeatures1; i++ ) {
      feat = feat1 + i;
      k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
      if( k == 2 ) {
	  d0 = descr_dist_sq( feat, nbrs[0] );
	  d1 = descr_dist_sq( feat, nbrs[1] );
	  if( d0 < d1 * NN_SQ_DIST_RATIO_THR ) {
	      m++;
	      feat1[i].fwd_match = nbrs[0];
	  }
      }
      free( nbrs );
  }

  if (DEBUG)
    printf("computing transform\n");


  // Compute homography
  CvMat* H;
  H = ransac_xform(feat1, numFeatures1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
		   homog_xfer_err, 3.0, NULL, NULL);

  if (DEBUG)
    printf("past ransac_xform\n");

  if (!H) {
    if (DEBUG)
      printf("setting empty homography...\n");
    H = cvCreateMat(3, 3, CV_64F);
    for (x = 0; x < 3; x++) {
      for (y = 0; y < 3; y++) {
	cvmSet(H, x, y, 1000000.0);
      }
    }
  }

  kdtree_release(kd_root);

  if (DEBUG)
    printf("setting homography\n");
  for (x = 0; x < 3; x++) {
    for (y = 0; y < 3; y++) {
      double tempValue = cvmGet(H, x, y);
      cvmSet(homography, x, y, tempValue);
    }
  }
  if (DEBUG)
    printf("past setting homography\n");

  free(feat1);
  free(feat2);

  pthread_exit(NULL);

}

// Print Match/Homography/Matrix
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

// ***************************CHECKPOINT 5 Methods: Calculating Norms of Homographies*******
// Calculate similarity of homographies
void calculateNorms(double* confidences, CvMat** matches, int image_count, int match_count)
{
  int i, j;
  
  CvMat* identity = create3DIdentity();  
  for (i = 0; i < match_count; i++) {
    if (!matches[i]) continue;
    confidences[i] = cvNorm(matches[i], identity, CV_L2, 0);
    //    printf("\nIndex is: %d\n", i);
    // printMatrix(matches[i]);
    //printf("\n");
  }

  //  if (DEBUG) {
    printf("\n");
    for (i = 0; i < image_count; i++) {
      for (j = 0; j < image_count; j++) {
	//printf("%.2f\t", confidences[i*image_count + j]);
      }
      //printf("\n");
    }
    //}
}

// **********************CHECKPOINT 6 METHODS: Generating Best Homographies*****************
// Find best homography - Step 4 of Algorithm:
void generateBestHomographies(int* bestMatchedIndex, int image_count, double* confidences) 
{
  int i, j, minConfidenceIndex;
  long double minConfidence;
  for (i = 0; i < image_count; i++) {
    minConfidence = 100000000000000000.0;
    minConfidenceIndex = 0;
    for (j = 0; j < image_count; j++) {
      if (i == j) continue;
      double value = confidences[i*image_count + j];
      if (value == 0.0) continue;
      if (value < minConfidence) {
	minConfidence = value;
	minConfidenceIndex = j;
      }
    }
    bestMatchedIndex[i] = minConfidenceIndex;
  }
}

// ***********************Render Checkpoint METHODS: Rendering Scene************************
// Init Default Scene
void initScene(int index, struct scene* aScene, struct pData* poses, int image_count)
{
  // init poses to index
  aScene->pose = poses[index];
  aScene->pose.eye = aScene->pose.center;
  aScene->pose.eye.val[0] = aScene->pose.eye.val[0] + 1;
  aScene->pose.up = cvScalar(0.0, 0.0, 1.0, 0.0);

  // init indices to index
  aScene->currentIndex = index;
  aScene->previousIndex = index;

  // init max image
  aScene->max_image = image_count;

  // init angles to 0
  aScene->xAngle = 0.0;
  aScene->yAngle = 0.0;
  aScene->zAngle = 0.0;

  if (DEBUG) {
    printScene(aScene);
  }
}

// Find closest index to myScene's pose
int closestImageIndex(struct pData* poses)
{
  int image_count = myScene->max_image;
  int baseIndex = myScene->currentIndex;

  int tempCount;
  int minIndex = baseIndex;
  double minDistance = 100000000000000000.0;
  int range = 5;
  for (tempCount = -1*range; tempCount < range; tempCount++) {
    int index = (baseIndex + tempCount) % image_count;
    if (index < 0) index += image_count;

    CvScalar currCenter = poses[index].center;
    CvScalar diff = cvScalar(0.0, currCenter.val[1] - myScene->pose.center.val[1],
			     currCenter.val[2] - myScene->pose.center.val[2], 0.0);

    /*
    CvScalar one = cvScalar(poses[index].center.val[0] - poses[index].eye.val[0],
			    poses[index].center.val[1] - poses[index].eye.val[1],
			    poses[index].center.val[2] - poses[index].eye.val[2], 0.0);
    CvScalar two = cvScalar(myScene->pose.center.val[0] - myScene->pose.eye.val[0],
			    myScene->pose.center.val[1] - myScene->pose.eye.val[1],
			    myScene->pose.center.val[2] - myScene->pose.eye.val[2], 0.0);

    double direction = dot(one, two);
    */
    double tempNorm = norm(diff);

    if (tempNorm < minDistance) {
      minDistance = tempNorm;
      minIndex = index;
    }
  }

  if (minIndex != baseIndex) {
    myScene->previousIndex = myScene->currentIndex;
    myScene->currentIndex = minIndex;
    baseIndex = minIndex;

    if (DEBUG) {
      printf("minIndex = %d\n", minIndex);
      printPose(poses[minIndex]);
      printPose(myScene->pose);
      printf("\n");
    }

  } else {
    myScene->previousIndex = myScene->currentIndex;
  }

  return baseIndex;
}

// Calculate cameraHomography -- go from current position to camera position
CvMat* modelViewMatrix(int baseIndex, struct pData* poses)
{
  // Initial Homography
  CvMat* initHomography = create3DIdentity();

  // Find Forward and Up Vector
  CvScalar forward = cvScalarAll(0);
  createForwardVector(&forward, myScene->pose, poses[baseIndex]);
  CvScalar up = poses[baseIndex].up;
  //printf("forward vector: [%.2lf %.2lf %.2lf]\n", forward.val[0], forward.val[1], forward.val[2]);
  //printf("up vector: [%.2lf %.2lf %.2lf]\n", up.val[0], up.val[1], up.val[2]);

  // the z-axis
  double forwardAngle = atan2(forward.val[1], forward.val[0]);
  if (forwardAngle < 0)
    forwardAngle += PI;
  //printf("forwardAngle: %.2lf\n", forwardAngle * 180 / PI);

  // the y-axis
  double upAngleY = atan2(up.val[0], up.val[2]);
  //  if (upAngleY < 0)
  //  upAngleY += PI;
  //printf("upAngleY: %.2lf\n", upAngleY * 180 / PI);

  // the x-axis
  double upAngleX = atan2(up.val[1], up.val[2]);
  //if (upAngleX < 0)
  //  upAngleX += PI;
  //printf("upAngleX: %.2lf\n", upAngleX * 180 / PI);

  CvMat* rotateXHomography = cvCreateMat(3, 3, CV_64F);
  makeXAxisRotation(rotateXHomography, upAngleX);

  //printf("X: \n");
  //printMatrix(rotateXHomography);

  CvMat* rotateYHomography = cvCreateMat(3, 3, CV_64F);
  makeYAxisRotation(rotateYHomography, upAngleY);

  //printf("Y: \n");
  //printMatrix(rotateYHomography);

  CvMat* rotateZHomography = cvCreateMat(3, 3, CV_64F);
  makeZAxisRotation(rotateZHomography, forwardAngle);
  

  //printf("Z: \n");
  //printMatrix(rotateZHomography);

  // Apply all transformations
  cvMatMulAdd(rotateXHomography, initHomography, 0, initHomography);
  //cvMatMulAdd(rotateYHomography, initHomography, 0, initHomography);  
  //cvMatMulAdd(rotateZHomography, initHomography, 0, initHomography);
  projectTransform(initHomography);

  //printf("final: \n");
  //printMatrix(initHomography);

  cvReleaseMat(&rotateXHomography);
  cvReleaseMat(&rotateYHomography);
  cvReleaseMat(&rotateZHomography);

  return initHomography;

}

// Render Scene Struct
void renderScene(IplImage** images, CvMat** matches, int* bestMatchedIndex, 
		 IplImage* viewport, struct pData* poses)
{
  if (myScene->currentIndex != myScene->previousIndex) {
    //printf("CURRENT POSE: \n");
    //printPose(myScene->pose);
  }

  int i, j, k;
  int image_count = myScene->max_image;
  int baseIndex = closestImageIndex(poses);
  CvMat* transform = modelViewMatrix(baseIndex, poses);  

  // Translation?
  cvmSet(transform, 0, 2, -1*(myScene->pose.center.val[1] - poses[baseIndex].center.val[1]));
  cvmSet(transform, 1, 2, 1*(myScene->pose.center.val[2] - poses[baseIndex].center.val[2]));

  // Rotation?
  CvScalar diff = cvScalar(myScene->pose.center.val[0] - myScene->pose.eye.val[0], 
			   myScene->pose.center.val[1] - myScene->pose.eye.val[1], 
			   myScene->pose.center.val[2] - myScene->pose.eye.val[2], 0.0);
  //printf("diff is: [%.2lf  %.2lf  %.2lf  %.2lf]\n", diff.val[0], diff.val[1], diff.val[2], diff.val[3]);

  double radius = norm(diff);

  double angle1 = acos(diff.val[0] / radius) - PI;
  double angle2 = asin(diff.val[1] / radius);

  //printf("angle1: %.2lf\n", angle1);
  //printf("angle2: %.2lf\n", angle2);

  CvMat* zRotation = cvCreateMat(3, 3, CV_64F);
  makeZAxisRotation(zRotation, (angle1+angle2) / 2);
  //cvmSet(zRotation, 0, 2, 200*angle1);
  //cvmSet(zRotation, 1, 2, 200*angle1);
  cvMatMulAdd(zRotation, transform, 0, transform);
  cvReleaseMat(&zRotation);


  // Zoom?
  double zoom = radius;
  CvMat* zoomTransform = create3DIdentity();
  cvmSet(zoomTransform, 0, 0, zoom);
  cvmSet(zoomTransform, 1, 1, zoom);
  cvMatMulAdd(zoomTransform, transform, 0, transform);
  cvReleaseMat(&zoomTransform);

  for (k = 0; k < MATCH_RANGE; k++) {
    i = (baseIndex + k) % image_count;
    if (i < 0) i += image_count;
    //printf("displaying image %d\n", i);

    if (i == baseIndex) {
      cvWarpPerspective(images[i], viewport, transform, CV_INTER_LINEAR, cvScalarAll(0));
      continue;
    }

    //mosaic other images
    mosaic(i, images, matches, bestMatchedIndex, viewport, transform);
  }
  cvReleaseMat(&transform);

}


// Render more than one image per frame -- mosaic
void mosaic(int index, IplImage** images, CvMat** matches, int* bestMatchedIndex, 
		 IplImage* viewport, CvMat* initHomography)
{
  int image_count = myScene->max_image;
  int baseIndex = myScene->currentIndex;
  int history[image_count];
  int j;

  CvMat *ident = create3DIdentity();
  for (j = 0; j < image_count; j++) {
    history[j] = -1;
  }
  
  CvMat* topHomography = copyMatrix(matches[index*image_count + baseIndex]);
  double topConfidence = topConfidence = cvNorm(topHomography, ident, CV_L2, 0);
  
  CvMat* currHomography = copyMatrix(matches[index*image_count + bestMatchedIndex[index]]);;
  int currIndex = bestMatchedIndex[index];
  
  while (currIndex != baseIndex) {
    if (!matches[currIndex*image_count + baseIndex]) {
      //printf("FIX ME!\n");
      currIndex = -1;
      break;
    }
    CvMat* tempHomography = cvCreateMat(3, 3, CV_64F);
    cvMatMul(matches[currIndex*image_count + baseIndex], 
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
  
  if (currIndex == baseIndex) {
    if (DEBUG) {
      printf("Index to Match: bestBaseImageIndex = %d\n", currIndex);
      printf("Chosen homography: ");
      printMatrix(currHomography);
    }
   
    // multiply initial homography
    cvMatMulAdd(initHomography, currHomography, 0, currHomography);
    cvWarpPerspective(images[index], viewport, currHomography, CV_INTER_LINEAR, cvScalarAll(0));
  } else {
    //if (DEBUG) {
      printf("Index to Match: currIndex = %d\n", currIndex);
      printf("Chosen homography: ");
      printMatrix(topHomography);
      //}
    
    // multiply initial homography
    cvMatMulAdd(initHomography, topHomography, 0, topHomography);
    cvWarpPerspective(images[index], viewport, topHomography, CV_INTER_LINEAR, cvScalarAll(0));
  }
  if (DEBUG){
    cvShowImage("Scene", viewport);
    cvWaitKey(0);
  }
  cvReleaseMat(&topHomography);
  cvReleaseMat(&currHomography);
  cvReleaseMat(&ident);  
}


// Update Scene Struct
void updateSceneKey(int key)
{

  //printf("key: %d\n", key);
  int temp = 0;
  int image_count = myScene->max_image;
  switch(key) {
    case 65361: // left
      if (myScene->currentIndex > 0) {
	temp = myScene->currentIndex;
	myScene->previousIndex = myScene->currentIndex;
	myScene->currentIndex = temp-1;
      } else {
	myScene->previousIndex = myScene->currentIndex;
      }
      break;
    case 65363: // right
      if (myScene->currentIndex < image_count - 1) {
	temp = myScene->currentIndex;
	myScene->previousIndex = myScene->currentIndex;
	myScene->currentIndex = temp+1;
      } else {
	myScene->previousIndex = myScene->currentIndex;
      }
      break;
    default:
      if (OPTION && myScene->currentIndex < image_count - 1) {
	temp = myScene->currentIndex;
	myScene->previousIndex = myScene->currentIndex;
	myScene->currentIndex = temp+1;	
      } else {      
	myScene->previousIndex = myScene->currentIndex;
      }
  }

}

// Update Scene Struct
void updateSceneMouse(double diffX, double diffY)
{
  double eyeX = myScene->pose.eye.val[0];
  double eyeY = myScene->pose.eye.val[1];
  double eyeZ = myScene->pose.eye.val[2];

  double centerX = myScene->pose.center.val[0];
  double centerY = myScene->pose.center.val[1];
  double centerZ = myScene->pose.center.val[2];
  
  double threshold = .1;
  double scale = 10;
  double maxAngle = PI/4;
  
  // Rotating from Z-Axis (optional X & Y) - move mouse left and right
  if (leftClicked && !rightClicked) {
      CvScalar diff = cvScalar(centerX - eyeX, 
			       centerY - eyeY,
			       centerZ - eyeZ, 0.0);
      double radius = norm(diff);
      
      myScene->pose.eye.val[0] = centerX + radius * cos(maxAngle * diffY);
      myScene->pose.eye.val[1] = centerY + radius * sin(maxAngle * diffY);
  }

  // Translation along Z-Y plane
  if (!leftClicked && rightClicked) {
    
    if ((absD(diffX) > threshold) && (absD(diffY) < threshold)) {
      myScene->pose.eye.val[1] = eyeY + scale*diffX;
      myScene->pose.center.val[1] = centerY + scale*diffX;

    } else if ((absD(diffX) < threshold) && (absD(diffY) > threshold)) {
      myScene->pose.eye.val[2] = eyeZ + scale*diffY;
      myScene->pose.center.val[2] = centerZ + scale*diffY;

    } 
  } 

  // Zooming along X-Axis - move mouse up and down
  if ((leftClicked && rightClicked) || middleClicked) {
    CvScalar diff = cvScalar(centerX - eyeX, 
			     centerY - eyeY,
			     centerZ - eyeZ, 0.0);
    double radius = norm(diff);
    
    double zoom = radius + diffX/10;
    
    myScene->pose.eye.val[0] = centerX + zoom * diff.val[0] / radius;
    myScene->pose.eye.val[1] = centerY + zoom * diff.val[1] / radius;
  }
}

// Print Scene Struct
void printScene(struct scene* aScene)
{
  printf("current index: %d\n", aScene->currentIndex);
  printf("previous index: %d\n", aScene->currentIndex);
  printPose(aScene->pose);
  printf("max image: %d\n", aScene->max_image);
}

// Scene Mouse Event Handler
void mouseHandler(int event, int x, int y, int flags, void* param)
{
  double xVal = 0.0, yVal = 0.0, zVal = 0.0;
  double xAngle = 0.0, yAngle = 0.0, zAngle = 0.0;
  double zoom = 0.0;
  switch(event) {
    case CV_EVENT_LBUTTONDOWN:// && !CV_EVENT_RBUTTONDOWN:
      leftClicked = 1;
      break;
    case CV_EVENT_LBUTTONUP:
      leftClicked = 0;
      break;
    case CV_EVENT_RBUTTONDOWN:// && !CV_EVENT_LBUTTONDOWN:
      rightClicked = 1;
      break;
    case CV_EVENT_RBUTTONUP:
      rightClicked = 0;
      break;
    case CV_EVENT_MBUTTONDOWN:
      middleClicked = 1;
      break;
    case CV_EVENT_MBUTTONUP:
      middleClicked = 0;
    case CV_EVENT_MOUSEMOVE:
      if (!rightClicked && !leftClicked && !middleClicked) {
	// Where the mouse was when last clicked
	mousePosX = ((double) x) / WINDOW_WIDTH;
	mousePosY = ((double) y) / WINDOW_HEIGHT;
      } else {
	double newPosX = ((double) x) / WINDOW_WIDTH;
	double newPosY = ((double) y) / WINDOW_HEIGHT;
	
	double diffX = newPosX - mousePosX;
	double diffY = mousePosY - newPosY;

	updateSceneMouse(diffX, diffY);
      }
      break;
    default:
      break;
  }
}

// ***************************Utility/Library Methods**************************************
// Temporary identity
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

CvMat* copyMatrix(CvMat* mat)
{
  int i, j;
  CvMat *temp = cvCreateMat(3, 3, CV_64F);
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      double val = cvmGet(mat, i, j);
      cvmSet(temp, i, j, val);
    }
  }
  return temp;
}

// Set a homography with values
void setHomography(CvMat *mat, double e00, double e01, double e02, 
		   double e10, double e11, double e12,
		   double e20, double e21, double e22)
{
  cvmSet(mat, 0, 0, e00);
  cvmSet(mat, 0, 1, e01);
  cvmSet(mat, 0, 2, e02);
  cvmSet(mat, 1, 0, e10);
  cvmSet(mat, 1, 1, e11);
  cvmSet(mat, 1, 2, e12);
  cvmSet(mat, 2, 0, e20);
  cvmSet(mat, 2, 1, e21);
  cvmSet(mat, 2, 2, e22);
}

// Print Pose Struct
void printPose(struct pData pose)
{
  printf("Eye Vector: ");
  printf("[%.2f %.2f %.2f]\n", pose.eye.val[0], pose.eye.val[1], pose.eye.val[2]);
  printf("Center Vector: ");
  printf("[%.2f %.2f %.2f]\n", pose.center.val[0], pose.center.val[1], pose.center.val[2]);
  printf("Up Vector: ");
  printf("[%.2f %.2f %.2f]\n", pose.up.val[0], pose.up.val[1], pose.up.val[2]); 
}


CvScalar createForwardVector(CvScalar* forward, struct pData dstPose, struct pData srcPose)
{
  CvScalar point =  cvScalar(srcPose.center.val[0] - srcPose.eye.val[0],
			     srcPose.center.val[1] - srcPose.eye.val[1],
			     srcPose.center.val[2] - srcPose.eye.val[2], 0.0);
  
  double dotProduct = dot(point, dstPose.up);

  forward->val[0] = point.val[0] - dotProduct * dstPose.up.val[0];
  forward->val[1] = point.val[1] - dotProduct * dstPose.up.val[1];
  forward->val[2] = point.val[2] - dotProduct * dstPose.up.val[2];

  if (DEBUG)
    printf("forward: x = %.2lf, y = %.2lf, z = %.2lf\n", forward->val[0], forward->val[1], forward->val[2]);

}

void makeXAxisRotation(CvMat* dst, double angle)
{
  setHomography(dst, 
		cos(angle), -1*sin(angle), 0.0,
		sin(angle), cos(angle), 0.0,
		0.0, 0.0, 1.0);


  if (DEBUG) {
    printf("X: \n");
    printMatrix(dst);
  }
}

void makeYAxisRotation(CvMat* dst, double angle)
{
  setHomography(dst, 
		1.0, 0.0, 0.0,
		0.0, cos(angle), -1*sin(angle),
		0.0, sin(angle), cos(angle));

  if (DEBUG) {
    printf("Y: \n");
    printMatrix(dst);
  }
}

// make rotation matrix given angle in radians
void makeZAxisRotation(CvMat* dst, double angle)
{

  setHomography(dst, 
		cos(angle), 0.0, sin(angle),
		0.0, 1.0, 0.0,
		-1*sin(angle), 0.0, cos(angle));

  if (DEBUG) {
    printf("Z: \n");
    printMatrix(dst);
  }

}

double dot(CvScalar one, CvScalar two)
{
  return one.val[0] * two.val[0] + one.val[1] * two.val[1] + one.val[2] * two.val[2];
}

double norm(CvScalar vec)
{
  return sqrt(pow(vec.val[0], 2) + pow(vec.val[1], 2) + pow(vec.val[2], 2));
}

void projectTransform(CvMat* dst)
{
  cvmSet(dst, 2, 0, 0.0);
  cvmSet(dst, 2, 1, 0.0);
  cvmSet(dst, 2, 2, 1.0);
  cvmSet(dst, 0, 2, 0.0);
  cvmSet(dst, 1, 2, 0.0);
}

double absD(double d)
{
  if (d < 0)
    return -1*d;
  else
    return d;
}
