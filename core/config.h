#ifndef TEST_CONFIG_H
#define TEST_CONFIG_H

//#define CPU_ONLY
#ifndef CPU_ONLY
  #define CPU_ID 0
#endif // CPU_ONLY

#define MODEL_DIR "models"

// ************ MTCNN *************
#define MTCNN_MODEL_DIR MODEL_DIR
#define MTCNN_MIN_SIZE 80
#define MTCNN_MAX_SIZE 200
#define MTCNN_SCALE_FACTOR 0.65
#define MTCNN_PNET_THRESHOLD 0.8
#define MTCNN_RNET_THRESHOLD 0.9
#define MTCNN_ONET_THRESHOLD 0.9
//#define MTCNN_PRECISE_LANDMARK

// ************ AMFACE *************
#define AMFACE_MODEL_DIR MODEL_DIR
#define AMFACE_BATCH_SIZE 50

#endif // TEST_CONFIG_H