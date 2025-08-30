/***********************************************************************************************************************
* Copyright (C) 2023 Renesas Electronics Corporation. All rights reserved.
***********************************************************************************************************************/
/***********************************************************************************************************************
* File Name    : define.h
* Version      : 0.90
* Description  : RZ/V2H DRP-AI Sample Application for Megvii-Base Detection YOLOv8 with MIPI/USB Camera
***********************************************************************************************************************/

#ifndef DEFINE_MACRO_H
#define DEFINE_MACRO_H

/*****************************************
* includes
******************************************/
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <vector>
#include <map>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <cstring>
#include <float.h>
#include <atomic>
#include <semaphore.h>
#include <numeric>
/*****************************************
* Macro for YOLOv8
******************************************/
/* Input Camera support */
/* n = 0: USB Camera, n = 1: eCAM22 */
#define INPUT_CAM_TYPE              (0)
#if INPUT_CAM_TYPE == 0
    #define CAM_INPUT_VGA
#else
    #define CAM_INPUT_FHD
#endif

/* Output Camera Size */
#define IMAGE_OUTPUT_FHD
#define MIPI_CAM_RES                "1920x1080"

/*Time Measurement Flag*/
//#define DEBUG_TIME_FLG

/*Display AI frame rate*/
#undef DISP_AI_FRAME_RATE

/* Padding input mode to maintain the aspect ratio of DRP-AI input image.
   This mode requires the DRP-AI object file having the squared input size CAM_IMAGE_WIDTH x CAM_IMAGE_WIDTH.
   0: No padding, 
   1: With padding (maintains the aspect ratio) */
#define DRPAI_INPUT_PADDING         (1)

/* Tuning of sigmoid timing for acceleration of CPU DFL and post processing.
   n = 0: Do sigmoid in DFL (Original implementation)
   n = 1: Skip sigmoid in DFL and do sigmoid after argmax in post processing. (Reduce the sigmoid time to 1/(NUM_CLASS))
   n = 2: Skip sigmoid in DFL and do sigmoid after threshold processing in post processing. (Reduce the sigmoid time to the number of the detected bounding box before NMS.)
   */ 
#define CPU_DFL_SIGMOID_SKIP        (2)

/* Enable or Disable the multi-threading for CPU DFL processing.
   n = 0: Disable (single-thread for CPU DFL)
   n = 1: Enable (multi-threads for CPU DFL)
   */ 
#define CPU_DFL_MULTI_THREAD        (1)

#if(1)  // TVM
/* DRP-AI memory offset for model object file*/
#define DRPAI_MEM_OFFSET            (0X38E0000)
#endif  // TVM

/*****************************************
* Static Variables for YOLOv8
* Following variables need to be changed in order to custormize the AI model
*  - drpai_prefix0 = directory name of DRP-AI Object files (DRP-AI Translator output)
******************************************/
#if(1)  // TVM
/* Model Binary */
const static std::string model_dir = "yolov8_cam";
/* Pre-processing Runtime Object */
const static std::string pre_dir = model_dir + "/preprocess";
#endif  // TVM

/*****************************************
* Macro for Yolov8
******************************************/
/* Number of class to be detected */
#define NUM_CLASS                   (80)
/* Number for [region] layer num parameter */
#define NUM_BB                      (1)
/* Number of output layers. This value MUST match with the length of num_grids[] below */
#define NUM_INF_OUT_LAYER           (3)
/* Number of grids in the image. The length of this array MUST match with the NUM_INF_OUT_LAYER */
const static uint8_t num_grids[] = { 80,40,20 };
/* Number of DFL channel (default:16) */
#define REG_MAX                     (16)

/* Number of DRP-AI output */
const static uint32_t num_dfl80_out = (REG_MAX * 4) * num_grids[0] * num_grids[0];
const static uint32_t num_dfl40_out = (REG_MAX * 4) * num_grids[1] * num_grids[1];
const static uint32_t num_dfl20_out = (REG_MAX * 4) * num_grids[2] * num_grids[2];
const static uint32_t num_class80_out = NUM_CLASS * num_grids[0] * num_grids[0];
const static uint32_t num_class40_out = NUM_CLASS * num_grids[1] * num_grids[1];
const static uint32_t num_class20_out = NUM_CLASS * num_grids[2] * num_grids[2];

const static uint32_t num_inf_out =  (NUM_CLASS + 4) * NUM_BB * num_grids[0] * num_grids[0]
                                   + (NUM_CLASS + 4) * NUM_BB * num_grids[1] * num_grids[1]
                                   + (NUM_CLASS + 4) * NUM_BB * num_grids[2] * num_grids[2];

const static uint32_t num_grid_points = num_grids[0] * num_grids[0] + num_grids[1] * num_grids[1] + num_grids[2] * num_grids[2];

/* Thresholds */
#define TH_PROB                     (0.5f)
#define TH_NMS                      (0.5f)
/* Size of input image to the model */
#define MODEL_IN_W                  (640)
#define MODEL_IN_H                  (640)

/*****************************************
* Macro for Application
******************************************/
/*Camera:: Capture Image Information*/
#ifdef CAM_INPUT_VGA
#define CAM_IMAGE_WIDTH             (640)
#define CAM_IMAGE_HEIGHT            (480)
#else /* CAM_INPUT_FHD */
#define CAM_IMAGE_WIDTH             (1920)
#define CAM_IMAGE_HEIGHT            (1080)
#endif

#define CAM_IMAGE_CHANNEL_YUY2      (2)
#define CAM_IMAGE_SIZE              (CAM_IMAGE_WIDTH * CAM_IMAGE_HEIGHT * CAM_IMAGE_CHANNEL_YUY2)

/*Camera:: Capture Information */
#if INPUT_CAM_TYPE == 1
#define CAP_BUF_NUM                 (6)
#define INPUT_CAM_NAME              "MIPI Camera"
#else /* INPUT_CAM_TYPE */
#define CAP_BUF_NUM                 (3)
#define INPUT_CAM_NAME              "USB Camera"
#endif /* INPUT_CAM_TYPE */

/*DRP-AI Input image information*/
#if (1) == DRPAI_INPUT_PADDING
/*** DRP-AI input is assigned to the buffer having the size of CAM_IMAGE_WIDTH^2 */
#define DRPAI_IN_WIDTH              (CAM_IMAGE_WIDTH)
#define DRPAI_IN_HEIGHT             (CAM_IMAGE_WIDTH) 
#define DRPAI_IN_CHANNEL_YUY2       (CAM_IMAGE_CHANNEL_YUY2)
#else
/** DRP-AI input is assigned to the buffer having the size of camera image. */
#define DRPAI_IN_WIDTH              (CAM_IMAGE_WIDTH)
#define DRPAI_IN_HEIGHT             (CAM_IMAGE_HEIGHT)  
#define DRPAI_IN_CHANNEL_YUY2       (CAM_IMAGE_CHANNEL_YUY2)
#endif

/*Wayland:: Wayland Information */
#ifdef IMAGE_OUTPUT_HD
#define IMAGE_OUTPUT_WIDTH          (1280)
#define IMAGE_OUTPUT_HEIGHT         (720)
#else /* IMAGE_OUTPUT_FHD */
#define IMAGE_OUTPUT_WIDTH          (1920)
#define IMAGE_OUTPUT_HEIGHT         (1080)
#endif

/*Camera image size displayed on HDMI image.*/
#ifdef CAM_INPUT_VGA
#define CAM_RESIZED_WIDTH        (CAM_IMAGE_WIDTH*2)
#define CAM_RESIZED_HEIGHT       (CAM_IMAGE_HEIGHT*2)
#else /* CAM_INPUT_FHD */
#define CAM_RESIZED_WIDTH        (IMAGE_OUTPUT_WIDTH)
#define CAM_RESIZED_HEIGHT       (IMAGE_OUTPUT_HEIGHT)
#endif

#define IMAGE_CHANNEL_BGRA          (4)
#define WL_BUF_NUM                  (2)

/*Image:: Text information to be drawn on image*/
#define CHAR_SCALE_LARGE            (0.8)
#define CHAR_SCALE_SMALL            (0.7)
#define CHAR_THICKNESS              (2)
#define LINE_HEIGHT                 (30) /*in pixel*/
#define LINE_HEIGHT_OFFSET          (20) /*in pixel*/
#define TEXT_WIDTH_OFFSET           (10) /*in pixel*/
#ifdef CAM_INPUT_VGA
#define CHAR_THICKNESS_BOX          (1)  /*in pixel*/
#define BOX_LINE_SIZE               (2)  /*in pixel*/
#define BOX_HEIGHT_OFFSET           (15) /*in pixel*/
#define BOX_TEXT_HEIGHT_OFFSET      (5)  /*in pixel*/
#define CHAR_SCALE_FONT             (0.5)
#else
#define CHAR_THICKNESS_BOX          (2)  /*in pixel*/
#define BOX_LINE_SIZE               (3)  /*in pixel*/
#define BOX_HEIGHT_OFFSET           (30) /*in pixel*/
#define BOX_TEXT_HEIGHT_OFFSET      (8)  /*in pixel*/
#define CHAR_SCALE_FONT             (0.8)
#endif
#define WHITE_DATA                  (0xFFFFFFu) /* in RGB */
#define BLACK_DATA                  (0x000000u)

/*Waiting Time*/
#define WAIT_TIME                   (1000) /* microseconds */

/*Timer Related*/
#define CAPTURE_TIMEOUT             (20)  /* seconds */
#define AI_THREAD_TIMEOUT           (20)  /* seconds */
#define DISPLAY_THREAD_TIMEOUT      (20)  /* seconds */
#define KEY_THREAD_TIMEOUT          (5)   /* seconds */
#define TIME_COEF                   (1)

/*Array size*/
#define SIZE_OF_ARRAY(array)        (sizeof(array)/sizeof(array[0]))

/*****************************************
* For image.cpp
******************************************/
/*For drawing the bounding box label on image*/
#define FONTDATA_WIDTH              (6)
#define FONTDATA_HEIGHT             (8)

#endif
